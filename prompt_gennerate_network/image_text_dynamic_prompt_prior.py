# analysis_and_ablation.py
"""
零样本CLIP4Cir基线实验：仅使用clip_model将文本和图像特征拼接后过一个线性层作为查询 得到静态权重，不是动态预测
输入: openadd.json (包含概念层次结构)
输出: concept_vectors.npy (向量数组), concept_names.json (概念名称列表)
依赖: pip install openai torch transformers numpy tqdm python-dotenv tenacity

基于clip4cir先验静态提示
质量偏低原因：
概念库与数据集不匹配：您的概念库（基于 ASAM OpenODD）可能偏通用，而 CityFlow-NL 中的描述更侧重于车辆外观、动作等细粒度属性。概念扩展描述未能覆盖查询中的关键语义。
概念向量质量：扩展描述可能由 LLM 生成，但与 CLIP 特征空间的对齐不够好，或者描述过于抽象，与真实场景的自然语言描述分布不同。
特征融合方式：您将文本特征和图像特征简单相加后归一化，这可能削弱了各自的有用信息，导致查询特征偏离有效语义。

纯先验改进策略：
改进概念库：
	检查概念库中的概念是否与 CityFlow-NL 数据集的描述语义匹配。例如，如果查询中出现“左转”，概念库中应有“转向”或“左转”等概念，且扩展描述应包含类似“车辆向左转弯”的句子。
	使用更强的 LLM（如 GPT-4）重新生成扩展描述，使其更贴近真实驾驶场景的描述方式。
	可以考虑不依赖扩展描述，直接使用概念名称向量（如“左转”、“黑色轿车”）计算相似度，有时名称向量反而更准确。
调整查询特征融合：
	当前采用 text_feat + img_feat 的简单相加可能削弱了各自的特征强度。尝试改为拼接后通过一个线性层（或 MLP）投影到相同维度，让模型学习融合权重。
	保留原始文本特征作为查询，不融合图像，看 prior 是否有改善（排除图像特征引入的噪声）。
验证概念向量质量：
	随机抽取一个概念（如“左转”），用 CLIP 文本编码器编码其名称和扩展描述，然后手动计算与几个典型查询文本的相似度，确认相似度是否合理。
暂时跳过 prior 相关实验：
	如果您的时间有限，可以先放弃基于 prior 的消融分析，直接尝试改进模型的其他部分（如参考图像融合方式、训练策略等）
"""
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入原始脚本中的所有必要组件
from image_text_dynamic_prompt_gennerate import (
    Config, clip_model, preprocess, clip_dim, get_or_create_track_split, load_concept_extensions, build_train_triplets,
    build_validation_data, ValidationDataset, compute_prior_scores, compute_prior_scores_single,
    evaluate_batched, PromptGenerator, TripletDataset, TrackMutualSampler
)

device = Config.device
concept_names, concept_name_embs, concept_extend_embs, concept_desc_embs = load_concept_extensions(
    Config.concept_extend_file, Config.concept_extend_embeddings
)


# ------------------------------------------------------------
# 1. 随机选取验证集中的数据分析 prior 矩阵统计信息，以观察 prior 得分是否具有区分度，不是训练的必要输入
# ------------------------------------------------------------
def analyze_prior_stats(val_dataset, sample_size=100):
    """
    从验证集中随机抽样，计算 prior 得分的统计信息。
    """
    clip_model.eval()
    queries = val_dataset.queries
    if len(queries) > sample_size:
        queries = random.sample(queries, sample_size)

    prior_scores = []
    for q in tqdm(queries, desc="Computing prior stats"):
        # 加载参考图像
        ref_img = Image.open(q['ref_img']).convert('RGB')
        ref_tensor = preprocess(ref_img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(q['caption']).to(device)
        with torch.no_grad():
            ref_feat = F.normalize(clip_model.encode_image(ref_tensor), dim=-1)
            text_feat = F.normalize(clip_model.encode_text(text_tokens), dim=-1)
            query_feat = F.normalize(text_feat + ref_feat, dim=-1)
        # prior = compute_prior_scores(query_feat, concept_extend_embs).cpu().squeeze().numpy()
        prior = compute_prior_scores_single(query_feat, concept_name_embs).cpu().squeeze().numpy()
        # prior = compute_prior_scores_single(query_feat, concept_desc_embs).cpu().squeeze().numpy()
        prior_scores.append(prior)

    prior_arr = np.stack(prior_scores)  # [num_queries, C]
    print("\n=== Prior Score Statistics ===")
    print(f"Mean: {prior_arr.mean():.4f}")
    print(f"Std:  {prior_arr.std():.4f}")
    print(f"Max:  {prior_arr.max():.4f}")
    print(f"Min:  {prior_arr.min():.4f}")
    print("(Per-query max prior: {:.4f})".format(prior_arr.max(axis=1).mean()))
    return prior_arr

# ------------------------------------------------------------
# 2. 纯先验（不使用 MLP）的检索性能
# ------------------------------------------------------------
class PriorOnlyRetriever(nn.Module):
    """仅使用先验得分生成动态提示，不经过 MLP"""
    def __init__(self, concept_name_embs, concept_extend_embs, temperature=0.07):
        super().__init__()
        self.concept_name_embs = concept_name_embs
        self.concept_extend_embs = concept_extend_embs
        self.temperature = temperature

    def forward(self, text_feat, img_feat=None):
        if img_feat is not None:
            query_feat = F.normalize(text_feat + img_feat, dim=-1)
        else:
            query_feat = text_feat
        prior = compute_prior_scores(query_feat, self.concept_extend_embs)
        weights = F.softmax(prior / self.temperature, dim=-1)
        dyn_prompt = weights @ self.concept_name_embs
        combined = text_feat + dyn_prompt
        return F.normalize(combined, dim=-1)

def evaluate_prior_only(val_dataset, temperature=0.07):
    """评估纯先验检索的 Recall@K 和 mAP"""
    retriever = PriorOnlyRetriever(concept_name_embs, concept_extend_embs, temperature).to(device)
    recalls, mAP = evaluate_batched(
        clip_model, retriever, val_dataset, device, temperature,
        batch_size=64, k_list=[1,5,10]
    )
    print("\n=== Prior-Only Retrieval Results ===")
    for k in [1,5,10]:
        print(f"Recall@{k}: {recalls[k]:.2f}%")
    print(f"mAP: {mAP:.2f}%")
    return recalls, mAP

# ------------------------------------------------------------
# 3. 训练仅使用 prior（不拼接 query_feat）的 PromptGenerator
#  学习的是根据 prior 得分重新校准概念权重，而不是直接从原始特征预测权重。
#  特征演变路径：prior [B, C] → Linear(C, hidden) → ReLU → Linear(hidden, C) → logits → weights → dyn_prompt → 增强文本特征
# ------------------------------------------------------------
class PriorOnlyMLPGenerator(nn.Module):
    """MLP输入只使用 prior，不拼接 query_feat"""
    def __init__(self, concept_name_embs, concept_extend_embs, num_concepts, hidden_dim=256):
        super().__init__()
        self.concept_name_embs = concept_name_embs
        self.concept_extend_embs = concept_extend_embs
        self.mlp = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, text_feat, img_feat=None):
        text_feat = text_feat.float()
        if img_feat is not None:
            img_feat = img_feat.float()
        if img_feat is not None:
            query_feat = F.normalize(text_feat + img_feat, dim=-1)
        else:
            query_feat = text_feat
        prior = compute_prior_scores(query_feat, self.concept_extend_embs)  # [B, C]
        logits = self.mlp(prior)
        weights = F.softmax(logits / self.temperature, dim=-1)
        dyn_prompt = weights @ self.concept_name_embs
        combined = text_feat + self.alpha * dyn_prompt
        return F.normalize(combined, dim=-1)

def train_prior_only_mlp(train_loader, val_dataset, num_epochs=5, lr=1e-5, temperature=0.07):
    """训练 PriorOnlyMLPGenerator 并评估"""
    num_concepts = concept_name_embs.size(0)
    model = PriorOnlyMLPGenerator(concept_name_embs, concept_extend_embs, num_concepts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_map = 0.0

    for epoch in range(1, num_epochs+1):
        print(f"\n--- PriorOnlyMLP Training Epoch {epoch}/{num_epochs} ---")
        # 自定义一个简化的训练循环（复用 train_epoch 逻辑但需适配）
        model.train()
        total_loss = 0
        num_batches = 0
        for ref_imgs, target_imgs, texts, _ in tqdm(train_loader, desc="Training"):
            ref_imgs, target_imgs, texts = ref_imgs.to(device), target_imgs.to(device), texts.to(device)
            bs = ref_imgs.size(0)
            with torch.no_grad():
                ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1).float()
                target_feat = F.normalize(clip_model.encode_image(target_imgs), dim=-1).float()
                text_feat = F.normalize(clip_model.encode_text(texts), dim=-1).float()
            query_feat = model(text_feat, ref_feat).float()
            logits = query_feat @ target_feat.T / temperature
            labels = torch.arange(bs, device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        print(f"Epoch {epoch} Loss: {total_loss/num_batches:.4f}")

        # 每轮结束后评估
        recalls, mAP = evaluate_batched(clip_model, model, val_dataset, device, temperature, batch_size=64)
        print(f"Validation -> R@1: {recalls[1]:.2f}%, mAP: {mAP:.2f}%")
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), "./checkpoints/prior_only_mlp_best.pth")
            print("Best model saved.")
    return best_map

@torch.no_grad()
def evaluate_clip_zeroshot(val_dataset, device, temperature=0.07, batch_size=64):
    clip_model.eval()
    # 候选特征
    candidate_feats = val_dataset.load_or_extract_candidate_features(clip_model, device).to(device)
    candidate_feats = candidate_feats.float()  # 统一为 float32
    queries = val_dataset.queries
    num_queries = len(queries)

    recalls = {1: 0, 5: 0, 10: 0}
    ap_sum = 0.0

    for start in tqdm(range(0, num_queries, batch_size), desc="Zero-shot CLIP"):
        end = min(start + batch_size, num_queries)
        batch_queries = queries[start:end]
        batch_size_actual = end - start

        # 提取 batch 内所有参考图像和文本的特征
        ref_imgs = []
        text_tokens_list = []
        target_idxss = []
        for q in batch_queries:
            ref_img = Image.open(q['ref_img']).convert('RGB')
            ref_tensor = preprocess(ref_img).unsqueeze(0).to(device)
            ref_imgs.append(ref_tensor)
            text_tokens = clip.tokenize(q['caption']).to(device)
            text_tokens_list.append(text_tokens)
            target_idxss.append(q['target_idxs'])

        ref_imgs = torch.cat(ref_imgs, dim=0)
        text_tokens = torch.cat(text_tokens_list, dim=0)

        # 提取特征
        ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1).float()
        text_feat = F.normalize(clip_model.encode_text(text_tokens), dim=-1).float()

        # 零样本融合方式1：相加（也可改为仅文本）
        query_feat = F.normalize(text_feat + ref_feat, dim=-1)  # [B, D]

        # 相似度矩阵
        sim = query_feat @ candidate_feats.T / temperature  # [B, C]

        # 对每个查询计算指标
        for i in range(batch_size_actual):
            sim_i = sim[i]
            sorted_indices = sim_i.argsort(descending=True)
            pos_idxs = target_idxss[i]
            P = len(pos_idxs)

            # ---- 计算 AP ----
            is_relevant = torch.zeros(len(candidate_feats), dtype=torch.bool, device=device)
            for idx in pos_idxs:
                is_relevant[idx] = True
            sorted_relevant = is_relevant[sorted_indices]
            hits = 0
            ap = 0.0
            for rank, rel in enumerate(sorted_relevant):
                if rel:
                    hits += 1
                    precision_at_k = hits / (rank + 1)
                    ap += precision_at_k
            if P > 0:
                ap /= P
            ap_sum += ap

            # ---- 计算 Recall@K 和 MRR（第一个正样本位置）----
            first_rank = None
            for rank_idx, idx in enumerate(sorted_indices.cpu().tolist()):
                if idx in pos_idxs:
                    first_rank = rank_idx
                    break
            if first_rank is not None:
                for k in recalls:
                    if first_rank < k:
                        recalls[k] += 1

    num_q = len(queries)
    print("\n===== Zero-shot CLIP (text+img) Evaluation =====")
    for k in sorted(recalls.keys()):
        print(f"Recall@{k}: {recalls[k] / num_q * 100:.2f}%")
    print(f"mAP: {ap_sum / num_q * 100:.2f}%")
    return recalls, ap_sum / num_q * 100
# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
def main():
    # 1. 构建验证集（与原始训练相同的划分方式）
    print("Loading data...")
    split_file = os.path.join(Config.save_dir, 'track_split.pkl')   # 保存到 checkpoints 目录下
    train_track_ids, val_track_ids = get_or_create_track_split(
        Config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )
    candidate_images, val_queries = build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=3, sample_print=False
    )
    val_dataset = ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(Config.save_dir, 'candidate_feats_clipzeroshort.pt')
    )
    print(f"Validation set size: {len(val_queries)} queries, {len(candidate_images)} candidates.")

    # 纯clip模型零样本检索
    evaluate_clip_zeroshot(val_dataset, device, temperature=Config.temperature)

    # # 2. 分析 prior 统计信息
    analyze_prior_stats(val_dataset, sample_size=200)
    #
    # # 3. 评估纯先验检索
    # prior_recalls, prior_map = evaluate_prior_only(val_dataset, temperature=Config.temperature)
    #
    # # 4. 训练 PriorOnlyMLPGenerator（5个epoch，低学习率）
    # # 需要先构建训练集（使用训练车辆）
    # train_triplets = build_train_triplets(Config.track_ann_file, Config.image_root, allowed_track_ids=train_track_ids)
    # # 为了节省时间，可以只采样一小部分训练数据
    # if len(train_triplets) > 5000:
    #     train_triplets = random.sample(train_triplets, 5000)
    # train_dataset = TripletDataset(train_triplets, preprocess)
    # sampler = TrackMutualSampler(train_triplets, batch_size=Config.batch_size, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=Config.num_workers, pin_memory=True)
    # print(f"Training with {len(train_triplets)} triplets, batch_size={Config.batch_size} (effective batch).")
    # best_map = train_prior_only_mlp(train_loader, val_dataset, num_epochs=5, lr=1e-5)
    #
    # print("\n=== Summary ===")
    # print(f"Prior-only Recall@1: {prior_recalls[1]:.2f}%")
    # print(f"Prior-only mAP: {prior_map:.2f}%")
    # print(f"PriorOnlyMLP best mAP after 5 epochs: {best_map:.2f}%")

if __name__ == "__main__":
    # 注意：由于原始脚本中 concept_extend_embs 等是 list of list，需要确保它们已加载
    # 导入时已经执行了 load_concept_extensions，这些变量已存在
    # 如果因为循环导入出现问题，可以将以下代码放在 main 开头
    # 但这里我们假设导入时所有全局变量已就绪
    main()