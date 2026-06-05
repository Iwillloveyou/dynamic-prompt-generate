# ablation_study.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 复用原脚本中的所有组件
from image_text_dynamic_prompt_gennerate import (
    Config, clip_model, preprocess, clip_dim,
    get_or_create_track_split, build_train_triplets, build_validation_data,
    TripletDataset, ValidationDataset, load_concept_extensions, compute_prior_scores, compute_prior_scores_single,
    train_epoch, evaluate_batched
)


# ==================== 带消融标志的模型包装 ====================
class AblationPromptGenerator(nn.Module):
    """
    支持消融实验的 PromptGenerator：
    - remove_knowledge_base: 去除领域知识提示库 → 直接使用图像特征，不加动态提示
    - remove_generator: 去除提示生成网络 → 使用固定权重（均匀分布）
    - remove_semantic_ext: 去除语义扩展模块 → 使用概念名称向量而非扩展描述
    - remove_concept_activation: 去除领域概念激活模块 → 使用恒等映射（动态提示 = 0）
    """
    def __init__(self, concept_name_embs, concept_extend_embs, clip_dim, num_concepts,
                 hidden_dim=256,
                 remove_knowledge_base=False,
                 remove_generator=False,
                 remove_semantic_ext=False,
                 remove_concept_activation=False):
        super().__init__()
        self.concept_name_embs = concept_name_embs
        self.concept_extend_embs = concept_extend_embs
        self.num_concepts = num_concepts
        self.clip_dim = clip_dim
        self.remove_knowledge_base = remove_knowledge_base
        self.remove_generator = remove_generator
        self.remove_semantic_ext = remove_semantic_ext
        self.remove_concept_activation = remove_concept_activation

        # 如果完全去除知识库，则不需要任何可学习参数
        if not remove_knowledge_base:
            # MLP 输入维度：先验得分 + 查询特征
            if remove_semantic_ext:
                # 使用名称向量，先验维度为 num_concepts
                self.mlp = nn.Sequential(
                    nn.Linear(num_concepts + clip_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_concepts)
                )
            else:
                # 使用扩展描述，先验维度也是 num_concepts
                self.mlp = nn.Sequential(
                    nn.Linear(num_concepts + clip_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_concepts)
                )
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.temperature = nn.Parameter(torch.tensor(0.07))
        else:
            # 去除知识库时，不需要任何额外参数
            self.alpha = nn.Parameter(torch.tensor(1.0))  # 保持图像特征不变

    def forward(self, text_feat, img_feat=None):
        text_feat = text_feat.float()
        if img_feat is not None:
            img_feat = img_feat.float()

        # 1. 查询特征融合
        if img_feat is not None:
            query_feat = F.normalize(text_feat + img_feat, dim=-1)
        else:
            query_feat = text_feat

        # 2. 根据消融设置返回不同结果
        if self.remove_knowledge_base:
            # 直接返回图像特征（原模型中 combined = img_feat + alpha*dyn_prompt，但这里 dyn_prompt=0）
            return query_feat, None

        # 计算先验得分
        if self.remove_semantic_ext:
            # 使用概念名称向量（单向量相似度）
            prior = compute_prior_scores_single(query_feat, self.concept_name_embs)
        else:
            # 使用扩展描述（最大相似度）
            prior = compute_prior_scores(query_feat, self.concept_extend_embs)

        if self.remove_generator:
            # 不使用 MLP，直接用先验得分作为权重
            weights = F.softmax(prior / self.temperature, dim=-1)
        else:
            # 使用 MLP 预测权重
            mlp_input = torch.cat([prior, query_feat], dim=-1)
            logits = self.mlp(mlp_input)
            weights = F.softmax(logits / self.temperature, dim=-1)

        if self.remove_concept_activation:
            # 动态提示为 0
            dyn_prompt = torch.zeros_like(query_feat)
        else:
            dyn_prompt = weights @ self.concept_name_embs

        # 融合：图像特征 + 动态提示（根据原模型最新修改）
        combined = query_feat + self.alpha * dyn_prompt
        combined = F.normalize(combined, dim=-1)
        return combined, weights


def train_ablation_model(config, model_name, ablation_flags, train_track_ids, val_track_ids, track_to_int):
    """
    训练单个消融模型
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"Ablation flags: {ablation_flags}")
    print(f"{'='*50}")

    # 构建数据
    train_triplets = build_train_triplets(config.track_ann_file, config.image_root,
                                          allowed_track_ids=train_track_ids)
    train_dataset = TripletDataset(train_triplets, preprocess, track_to_int)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)

    candidate_images, val_queries = build_validation_data(
        config.track_ann_file, config.image_root, val_track_ids, num_targets=3,
        cache_file=os.path.join(config.save_dir, f'{model_name}_validation_cache.pkl')
    )
    val_dataset = ValidationDataset(candidate_images, val_queries, preprocess,
                                    cache_path=os.path.join(config.save_dir, f'{model_name}_candidate_feats.pt'))

    # 加载概念库
    concept_names, concept_name_embs, concept_extend_embs, _ = load_concept_extensions(
        config.concept_extend_file, config.concept_extend_embeddings
    )

    # 创建模型
    generator = AblationPromptGenerator(
        concept_name_embs, concept_extend_embs, clip_dim, len(concept_names),
        hidden_dim=config.hidden_dim,
        **ablation_flags
    ).to(config.device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_map = 0.0
    best_model_path = os.path.join(config.save_dir, f'{model_name}_best.pth')

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(clip_model, generator, train_loader, optimizer,
                                 config.device, config.temperature)
        if epoch % 5 == 0:
            recalls, mAP = evaluate_batched(clip_model, generator, val_dataset,
                                            config.device, config.temperature)
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, R@1={recalls[1]:.2f}%, mAP={mAP:.2f}%")
            if mAP > best_map:
                best_map = mAP
                torch.save(generator.state_dict(), best_model_path)
                print(f"Best model saved (mAP={best_map:.2f}%)")
        scheduler.step()

    print(f"Finished {model_name}, best mAP: {best_map:.2f}%")
    return best_map, best_model_path


def main():
    config = Config()
    device = config.device

    # 1. 获取数据划分
    split_file = os.path.join(config.save_dir, 'track_split.pkl')
    train_track_ids, val_track_ids = get_or_create_track_split(
        config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )
    print(f"Train tracks: {len(train_track_ids)}, Val tracks: {len(val_track_ids)}")

    # 构建 track_to_int 映射
    train_triplets_for_map = build_train_triplets(config.track_ann_file, config.image_root,
                                                  allowed_track_ids=train_track_ids)
    all_track_ids = sorted(set([t['track_id'] for t in train_triplets_for_map]))
    track_to_int = {tid: idx for idx, tid in enumerate(all_track_ids)}

    # 2. 定义消融实验配置
    ablation_configs = {
        "完整模型": {
            "remove_knowledge_base": False,
            "remove_generator": False,
            "remove_semantic_ext": False,
            "remove_concept_activation": False
        },
        "去除领域知识提示库": {
            "remove_knowledge_base": True,
            "remove_generator": False,
            "remove_semantic_ext": False,
            "remove_concept_activation": False
        },
        "去除提示生成网络": {
            "remove_knowledge_base": False,
            "remove_generator": True,
            "remove_semantic_ext": False,
            "remove_concept_activation": False
        },
        "去除语义扩展模块": {
            "remove_knowledge_base": False,
            "remove_generator": False,
            "remove_semantic_ext": True,
            "remove_concept_activation": False
        },
        "去除领域概念激活模块": {
            "remove_knowledge_base": False,
            "remove_generator": False,
            "remove_semantic_ext": False,
            "remove_concept_activation": True
        }
    }

    # 3. 运行消融实验
    results = {}
    for name, flags in ablation_configs.items():
        print(f"\n{'#'*60}\n# Starting: {name}\n{'#'*60}")
        best_map, model_path = train_ablation_model(config, name.replace(" ", "_"), flags,
                                                    train_track_ids, val_track_ids, track_to_int)
        results[name] = best_map

    # 4. 汇总结果
    print("\n" + "="*60)
    print("Ablation Study Results (mAP %):")
    print("="*60)
    full_model_map = results.get("完整模型", 0)
    for name, mAP in results.items():
        if name == "完整模型":
            print(f"{name}: {mAP:.2f}%")
        else:
            delta = mAP - full_model_map
            print(f"{name}: {mAP:.2f}% (Δ = {delta:+.2f}%)")

    # 5. 保存结果到 CSV
    df = pd.DataFrame(list(results.items()), columns=["Model", "mAP (%)"])
    df.to_csv(os.path.join(config.save_dir, "ablation_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(config.save_dir, 'ablation_results.csv')}")

    # 6. 绘制柱状图
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    maps = list(results.values())
    bars = plt.bar(models, maps, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    plt.ylabel('mAP (%)')
    plt.title('Ablation Study Results')
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, maps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, 'ablation_results.png'), dpi=300)
    plt.show()
    print(f"Plot saved to {os.path.join(config.save_dir, 'ablation_results.png')}")


if __name__ == "__main__":
    main()