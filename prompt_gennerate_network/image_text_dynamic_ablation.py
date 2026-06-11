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
    def __init__(self, concept_name_embs, concept_extend_embs, concept_extend_mean_embs, clip_dim, num_concepts,
                 hidden_dim=256,
                 remove_knowledge_base=False,
                 remove_generator=False,
                 remove_semantic_ext=False,
                 remove_concept_activation=False):
        super().__init__()
        self.concept_name_embs = concept_name_embs
        self.concept_extend_embs = concept_extend_embs
        self.concept_extend_mean_embs = concept_extend_mean_embs
        self.num_concepts = num_concepts
        self.clip_dim = clip_dim
        self.remove_knowledge_base = remove_knowledge_base
        self.remove_generator = remove_generator
        self.remove_semantic_ext = remove_semantic_ext
        self.remove_concept_activation = remove_concept_activation
        self.res_scale = nn.Parameter(torch.FloatTensor([0.1]))
        self.gamma_list = []

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

        self.gate_layer = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim),  # 接收图像和文本拼接特征
            nn.ReLU(),
            nn.Linear(clip_dim, 1),        # 输出一个标量门控值
            nn.Sigmoid()              # 限制在 0 ~ 1 之间
        )
        # 如果要走你之前设想的“文本主导增强”，我们可以加一个重映射层
        self.feature_combiner = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.LayerNorm(clip_dim)
        )
        # 🛠️ 执行恒等初始化：让矩阵变成单位矩阵，偏置清零
        nn.init.eye_(self.feature_combiner[0].weight)
        nn.init.zeros_(self.feature_combiner[0].bias)

    # 🛠️ 新增方法：清空上一轮的 gamma 记录
    def reset_gamma_tracking(self):
        self.gamma_list = []

    # 🛠️ 新增方法：计算整轮 Epoch 的 Gamma 平均值
    def get_average_gamma(self):
        if not self.gamma_list:
            return 0.0
        # 将所有 Batch 的 gamma 拼接并求全局平均值
        return torch.cat(self.gamma_list, dim=0).mean().item()

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
        if self.remove_concept_activation:
            # 去除概念激活时，让所有概念权重均分即可（均匀分布），保持稳定
            weights = torch.full((query_feat.size(0), self.num_concepts), 1.0 / self.num_concepts, device=query_feat.device)
        else:
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

        # 计算先验得分
        if self.remove_semantic_ext:
            dyn_prompt = weights @ self.concept_name_embs
        else:
            dyn_prompt = weights @ self.concept_extend_mean_embs
        dyn_prompt_norm = F.normalize(dyn_prompt, p=2, dim=-1)

        # 通过一个线性层学习门控信号
        gating_feat = torch.cat([img_feat, text_feat], dim=-1)
        gamma = torch.sigmoid(self.gate_layer(gating_feat)) # 输出在 0~1 之间
        self.gamma_list.append(gamma.detach().cpu())

        # 6. 融合回原始文本特征
        # combined = text_feat + self.alpha * dyn_prompt
        # 将动态提示与图像特征融合
        # combined = img_feat + self.alpha * dyn_prompt
        combined = gamma * img_feat + (1 - gamma) * dyn_prompt_norm
        fusion_res = self.feature_combiner(combined) # 映射平滑
        combined = F.normalize(combined + self.res_scale * fusion_res, dim=-1)
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
        cache_file=os.path.join(config.save_dir, f'clip4cir_validation_cache.pkl')
    )
    val_dataset = ValidationDataset(candidate_images, val_queries, preprocess,
                                    cache_path=os.path.join(config.save_dir, f'candidate_feats_clip4cir.pt'))

    # 加载概念库
    concept_names, concept_name_embs, concept_extend_embs, concept_desc_embs, concept_extend_mean_embs = load_concept_extensions(
        config.concept_extend_file, config.concept_extend_embeddings
    )

    # 创建模型
    generator = AblationPromptGenerator(
        concept_name_embs, concept_extend_embs, concept_extend_mean_embs, clip_dim, len(concept_names),
        hidden_dim=config.hidden_dim,
        **ablation_flags
    ).to(config.device)

    # optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)
    # 先取出要单独设置 lr 的参数
    skip_ids = {id(p) for p in list(generator.feature_combiner.parameters()) + list(generator.gate_layer.parameters())}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in generator.parameters() if id(p) not in skip_ids], 'lr': config.lr},
        {'params': generator.feature_combiner.parameters(), 'lr': 1e-4},
        {'params': generator.gate_layer.parameters(), 'lr': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_map = 0.0
    best_model_path = os.path.join(config.save_dir, f'{model_name}_gate2_best.pth')
    patience = 3
    early_stop_count = 0
    # 断点文件路径
    ckpt_path = os.path.join(Config.save_dir, f'resume_{model_name}_gate2_best.pth')
    start_epoch = 1

    # ========= 加载断点：存在就恢复所有训练状态 =========
    if os.path.exists(ckpt_path):
        # 解决你之前报错weights_only，加weights_only=False
        checkpoint = torch.load(ckpt_path, map_location=Config.device)
        generator.load_state_dict(checkpoint["combiner"])
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["sch"])
        start_epoch = checkpoint["epoch"] + 1  # 中断epoch跑完了，下一轮+1开始
        best_map = checkpoint["best_map"]
        early_stop_count = checkpoint["stop_cnt"]
        print(f"✅ 断点加载成功，从Epoch {start_epoch} 继续训练，历史best_mAP={best_map:.4f}")

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        generator.reset_gamma_tracking()
        train_loss = train_epoch(clip_model, generator, train_loader, optimizer,
                                 config.device, config.temperature)
        avg_gamma = generator.get_average_gamma()
        if avg_gamma > 0.0: # 规避消融了知识库导致没算gamma的情况
            print(f"📊 [Gamma Monitor] Image Weight (Gamma): {avg_gamma:.4f}")
        print(f"Train Loss: {train_loss:.4f}")
        # torch.save(generator.state_dict(), os.path.join(config.save_dir, f'train_{model_name}_best.pth'))

        # ========= 每轮训练完保存完整断点（断电/手动中断都能续） =========
        save_dict = {
            "combiner": generator.state_dict(),
            "opt": optimizer.state_dict(),
            "sch": scheduler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "stop_cnt": early_stop_count
        }
        torch.save(save_dict, ckpt_path)

        # 每 2 个 epoch 验证一次
        if epoch % 2 == 0:
            recalls, mAP, ndcg_sum = evaluate_batched(clip_model, generator, val_dataset,
                                                      config.device, config.temperature)
            print(f"Validation Results: R@1={recalls[1]:.2f}, R@5={recalls[5]:.2f}, R@10={recalls[10]:.2f}, MRR={mAP:.2f}")
            print(f"NDCG@5: {ndcg_sum[5]:.2f}%, NDCG@10: {ndcg_sum[10]:.2f}%")
            # mrr是什么指标，如果要计算map，该怎么修改
            if mAP > best_map:
                best_map = mAP
                early_stop_count = 0
                torch.save(generator.state_dict(), best_model_path)
                print(f"Best model saved (mAP={best_map:.2f}%)")
            else:
                early_stop_count += 1
                print(f"No improve, early_stop count: {early_stop_count}/{patience}")

        # 早停判断
        if early_stop_count >= patience:
            print(f"Early Stop Trigger! {patience} epochs no mAP improve, exit training.")
            break

        scheduler.step()
        print("Training finished.")

    print(f"Finished {model_name}, best mAP: {best_map:.2f}%")
    return best_map, best_model_path


def main(ablation_configs):
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

    # 3. 运行消融实验
    results = {}
    for name, flags in ablation_configs.items():
        print(f"\n{'#'*60}\n# Starting: {name}\n{'#'*60}")
        best_map, model_path = train_ablation_model(config, name.replace(" ", "_"), flags,
                                                    train_track_ids, val_track_ids, track_to_int)
        results[name] = best_map

if __name__ == "__main__":
    # 2. 定义消融实验配置
    ablation_configs = {
        # "完整模型": {
        #     "remove_knowledge_base": False,
        #     "remove_generator": False,
        #     "remove_semantic_ext": False,
        #     "remove_concept_activation": False
        # }
        # "去除领域知识提示库": {
        #     "remove_knowledge_base": True,
        #     "remove_generator": False,
        #     "remove_semantic_ext": False,
        #     "remove_concept_activation": False
        # },
        # "去除提示生成网络": {
        #     "remove_knowledge_base": False,
        #     "remove_generator": True,
        #     "remove_semantic_ext": False,
        #     "remove_concept_activation": False
        # },
        "去除语义扩展模块": {
            "remove_knowledge_base": False,
            "remove_generator": False,
            "remove_semantic_ext": True,
            "remove_concept_activation": False
        },
        # "去除领域概念激活模块": {
        #     "remove_knowledge_base": False,
        #     "remove_generator": False,
        #     "remove_semantic_ext": False,
        #     "remove_concept_activation": True
        # }
    }
    main(ablation_configs)