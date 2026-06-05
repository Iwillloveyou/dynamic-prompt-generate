#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CoCoOp 训练脚本（适配 CityFlow-NL 数据集）
使用可学习实例条件提示生成查询特征
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import random
import clip

# 添加父目录路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import image_text_dynamic_prompt_gennerate as pg

# ---------- 全局配置 ----------
device = pg.Config.device
clip_model = pg.clip_model
preprocess = pg.preprocess
clip_dim = pg.clip_dim

class Config:
    save_dir = pg.Config.save_dir
    track_ann_file = pg.Config.track_ann_file
    image_root = pg.Config.image_root
    num_targets = 3
    # 训练参数
    batch_size = 64
    epochs = 50
    lr = 1e-5
    temperature = 0.07
    num_workers = 4

# ---------- CoOp 实现 ----------
class CoOpTextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init="", dtype=torch.float32):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = dtype
        self.token_embedding = clip_model.token_embedding
        self.token_dim = self.token_embedding.embedding_dim
        self.max_seq_len = clip_model.positional_embedding.size(0)  # 通常是77

        if ctx_init:
            # 使用给定的词初始化
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            with torch.no_grad():
                ctx_ids = clip.tokenize(ctx_init).to(device).squeeze(0)[1:-1]  # 去掉SOS/EOS
                ctx_embeddings = self.token_embedding(ctx_ids).to(dtype)
            self.ctx = nn.Parameter(ctx_embeddings)
        else:
            self.ctx = nn.Parameter(torch.empty(n_ctx, self.token_dim, dtype=dtype))
            nn.init.normal_(self.ctx, std=0.02)
        self.n_ctx = n_ctx

    def forward(self, text_tokens):
        # text_tokens: [B, L] 其中 L = 77
        # 获取原始 token embeddings
        x = self.token_embedding(text_tokens).type(self.dtype)  # [B, L, D]

        # 替换策略：将前 n_ctx 个 token 的 embedding 替换为 ctx 向量
        # 注意：通常 text_tokens 的第一个 token 是 SOS（起始符），不应替换。因此我们从索引1开始替换
        # 假设模板为 [SOS, token1, token2, ..., tokenN, EOS, PAD...]
        # 我们将从索引1开始的 n_ctx 个 token 替换为 ctx
        ctx = self.ctx.unsqueeze(0).expand(text_tokens.size(0), -1, -1)  # [B, n_ctx, D]
        x[:, 1:1+self.n_ctx, :] = ctx

        self.clip_model = self.clip_model.float()
        # 序列长度保持不变，掩码无需修改
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D]
        x = self.clip_model.ln_final(x).type(self.dtype)

        # 取 EOS 位置的特征（通常 EOS 在原始序列中的位置不变）
        # 简化：取最后一个 token（但可能不是 EOS）。更准确的做法是找到每个样本的 EOS 索引
        eos_positions = text_tokens.argmax(dim=-1)  # 找到EOS token位置（CLIP中EOS id为49407）
        x = x[torch.arange(x.size(0)), eos_positions] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1)


class CoOpRetriever(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init=""):
        super().__init__()
        self.text_encoder = CoOpTextEncoder(clip_model, n_ctx, ctx_init)

    def forward(self, text_tokens, img_feat=None):
        # CoOp 不使用图像特征
        return self.text_encoder(text_tokens)

# ---------- CoCoOp 实现 ----------
class MetaNet(nn.Module):
    """元网络：从图像特征生成条件向量"""
    def __init__(self, clip_dim, n_ctx, token_dim):
        super().__init__()
        self.n_ctx = n_ctx
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )

    def forward(self, img_feat):
        # img_feat: [B, D]
        ctx = self.proj(img_feat)               # [B, token_dim]
        ctx = ctx.unsqueeze(1).expand(-1, self.n_ctx, -1)  # [B, n_ctx, token_dim]
        return ctx


class CoCoOpTextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init=""):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = torch.float32
        self.n_ctx = n_ctx
        token_embedding = clip_model.token_embedding
        self.token_dim = token_embedding.embedding_dim
        self.max_seq_len = clip_model.positional_embedding.size(0)   # 77

        self.meta_net = MetaNet(clip_dim, n_ctx, self.token_dim)

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            ctx_words = ctx_init.split(" ")
            n_ctx = len(ctx_words)
            with torch.no_grad():
                ctx_ids = clip.tokenize(ctx_init).to(device).squeeze(0)[1:-1]
                ctx_embeddings = token_embedding(ctx_ids).to(self.dtype)
            self.ctx_base = nn.Parameter(ctx_embeddings)
        else:
            self.ctx_base = nn.Parameter(torch.empty(n_ctx, self.token_dim, dtype=self.dtype))
            nn.init.normal_(self.ctx_base, std=0.02)
        self.n_ctx = n_ctx

    def forward(self, text_tokens, img_feat):
        x = self.clip_model.token_embedding(text_tokens).type(self.dtype)
        # 生成条件向量并与基向量相加
        ctx_cond = self.meta_net(img_feat)               # [B, n_ctx, D]
        ctx = ctx_cond + self.ctx_base.unsqueeze(0)      # [B, n_ctx, D]

        # 复制一份用于修改
        x_new = x.clone()  # 保留原始副本（实际不需要，直接修改 x 也可以，但为了清晰）

        # 步骤1：将原始文本（从索引1开始）向右平移 n_ctx 个位置
        # 原始序列: [SOS, t1, t2, ..., t_{L-2}, EOS, PAD...]  索引: 0..76
        # 平移后: 前 n_ctx 位置留给 ctx，原始 token 从索引 n_ctx+1 开始
        x_new[:, 1 + self.n_ctx : 77] = x[:, 1 : 77 - self.n_ctx]

        # 步骤2：将 ctx 放入前面的 n_ctx 个位置（从索引1开始）
        x_new[:, 1 : 1 + self.n_ctx] = ctx

        # 添加位置编码
        self.clip_model = self.clip_model.float()
        x_new = x_new + self.clip_model.positional_embedding.type(self.dtype)

        # Transformer
        x_new = x_new.permute(1, 0, 2)   # [L, B, D]
        x_new = self.clip_model.transformer(x_new)
        x_new = x_new.permute(1, 0, 2)   # [B, L, D]
        x_new = self.clip_model.ln_final(x_new).type(self.dtype)

        # 取 EOS token（原始 EOS 位置向右平移了 n_ctx）
        eos_positions = text_tokens.argmax(dim=-1) + self.n_ctx
        # 边界检查：确保 eos_positions 不超过 76
        eos_positions = torch.clamp(eos_positions, max=76)
        x_new = x_new[torch.arange(x_new.size(0)), eos_positions] @ self.clip_model.text_projection
        return F.normalize(x_new, dim=-1)


class CoCoOpRetriever(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init=""):
        super().__init__()
        self.text_encoder = CoCoOpTextEncoder(clip_model, n_ctx, ctx_init)

    def forward(self, text_tokens, img_feat):
        return self.text_encoder(text_tokens, img_feat)


# ---------- 训练函数 ----------
def train_epoch_cocoop(clip_model, retriever, train_loader, optimizer, device, temperature=0.07):
    """训练一个 epoch"""
    clip_model.eval()
    retriever.train()
    total_loss = 0
    num_batches = 0

    for ref_imgs, target_imgs, texts, track_ids in tqdm(train_loader, desc="Training"):
        ref_imgs = ref_imgs.to(device)
        target_imgs = target_imgs.to(device)
        texts = texts.to(device)
        batch_size = ref_imgs.size(0)

        with torch.no_grad():
            ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1).float()
            target_feat = F.normalize(clip_model.encode_image(target_imgs), dim=-1).float()
            text_feat = F.normalize(clip_model.encode_text(texts), dim=-1).float()
            # 注意：CoCoOp 需要原始的 text_tokens，不能直接用 text_feat
            # 所以我们不能在 with torch.no_grad() 里直接获取 text_feat
            # 需要重新获取 text_tokens
        text_tokens = texts  # texts 已经是 tokenized 的 ids

        # 生成查询特征（CoCoOp 使用图像特征和文本 tokens）
        query_feat = retriever(text_tokens, ref_feat)

        # 对比损失
        # logits = query_feat @ target_feat.T / temperature
        # labels = torch.arange(batch_size, device=device)
        # loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        # 使用多正样本对比损失
        track_ids = track_ids.to(device)   # 移动标签到 GPU
        loss = pg.multi_positive_contrastive_loss(query_feat, target_feat, track_ids, temperature)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ---------- 评估函数 ----------
@torch.no_grad()
def evaluate_retriever(retriever, val_dataset, device, temperature=0.07, batch_size=64):
    retriever.eval()
    clip_model.eval()

    # 候选图像特征
    candidate_feats = val_dataset.load_or_extract_candidate_features(clip_model, device).to(device)
    candidate_feats = candidate_feats.float()

    queries = val_dataset.queries
    num_queries = len(queries)

    recalls = {1: 0, 5: 0, 10: 0}
    ap_sum = 0.0

    for start in tqdm(range(0, num_queries, batch_size), desc="Evaluating"):
        end = min(start + batch_size, num_queries)
        batch_queries = queries[start:end]

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

        # 提取参考图像特征（所有方法都需要，CoOp 不用但为了接口统一仍计算）
        ref_feats = F.normalize(clip_model.encode_image(ref_imgs), dim=-1).float()

        # 调用 retriever 生成查询特征
        if isinstance(retriever, CoCoOpRetriever):
            query_feat = retriever(text_tokens, ref_feats)
        else:  # CoOp
            query_feat = retriever(text_tokens, None)

        # 相似度计算
        sim = query_feat @ candidate_feats.T / temperature

        for i in range(len(batch_queries)):
            # sim_i = sim[i]
            sim_i = sim[i].clone() # 克隆一行相似度，避免原地修改干扰并行计算
            # 【核心改动】：获取当前 Query 的参考图索引，强制将其相似度降到极低
            ref_idx = batch_queries[i].get('ref_idx')
            if ref_idx is not None:
                sim_i[ref_idx] = -1e9

            sorted_indices = sim_i.argsort(descending=True)
            pos_idxs = target_idxss[i]
            P = len(pos_idxs)

            # AP
            is_relevant = torch.zeros(len(candidate_feats), dtype=torch.bool, device=device)
            for idx in pos_idxs:
                is_relevant[idx] = True
            sorted_relevant = is_relevant[sorted_indices]
            hits = 0
            ap = 0.0
            for rank, rel in enumerate(sorted_relevant):
                if rel:
                    hits += 1
                    ap += hits / (rank + 1)
            if P > 0:
                ap /= P
            ap_sum += ap

            # Recall & MRR
            first_rank = None
            for rank, idx in enumerate(sorted_indices.cpu().tolist()):
                if idx in pos_idxs:
                    first_rank = rank
                    break
            if first_rank is not None:
                for k in recalls:
                    if first_rank < k:
                        recalls[k] += 1

    num_q = len(queries)
    print("\n=== Evaluation Results ===")
    for k in sorted(recalls.keys()):
        print(f"Recall@{k}: {recalls[k] / num_q * 100:.2f}%")
    real_mAP = (ap_sum / num_q) * 100
    print(f"mAP: {real_mAP:.2f}%")
    return recalls, real_mAP

def load_pretrained_weights(retriever, weights_path, device, method="coop"):
    """
    加载 CoOp/CoCoOp 官方预训练权重（适配 .pth.tar-50 格式）
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # 1. 提取模型参数字典（尝试常见 key）
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # 直接就是参数字典

    # 2. 移除可能的 'module.' 前缀（多卡训练时产生）
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 3. 打印几个原始键名示例，便于调试
    sample_keys = list(state_dict.keys())[:5]
    print(f"Original keys example: {sample_keys}")

    # 4. 为所有键添加 'text_encoder.' 前缀，因为我们的 retriever 将参数封装在 text_encoder 下
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[f"text_encoder.{k}"] = v

    # 5. 加载权重（strict=False 允许键不完全匹配）
    missing, unexpected = retriever.load_state_dict(new_state_dict, strict=False)

    if missing:
        print(f"Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys (first 10): {unexpected[:10]}")

    # 6. 验证是否加载了关键参数（例如 ctx）
    if hasattr(retriever.text_encoder, "ctx"):
        print(f"✅ ctx mean value after loading: {retriever.text_encoder.ctx.data.mean().item():.4f}")
    else:
        print("⚠️ Warning: text_encoder.ctx not found, loading may have failed.")

    print(f"✅ {method.upper()} weights loaded from {weights_path}")

# ---------- 使用预训练模型验证 ----------
# 使用预训练权重评估 CoOp
#python coop_cocoop_baseline.py --method coop --n_ctx 4 --weights /path/to/coop_vit_b32_ep50.pth

# 使用预训练权重评估 CoCoOp
#python coop_cocoop_baseline.py --method cocoop --n_ctx 4 --weights /path/to/cocoop_vit_b32_ep10.pth
def use_pre_train_model_val():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="coop", choices=["coop", "cocoop"])
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--n_ctx", type=int, default=4, help="Number of context tokens")
    parser.add_argument("--quick", action="store_true", help="Use small subset for testing")
    args = parser.parse_args()

    # 获取车辆划分（复用原脚本函数，若不存在则手动划分）
    split_file = os.path.join(Config.save_dir, 'track_split.pkl')
    if hasattr(pg, 'get_or_create_track_split'):
        train_track_ids, val_track_ids = pg.get_or_create_track_split(
            Config.track_ann_file, split_file, train_ratio=0.8, seed=42
        )
    else:
        random.seed(42)
        with open(Config.track_ann_file, 'r') as f:
            tracks = json.load(f)
        all_track_ids = list(tracks.keys())
        random.shuffle(all_track_ids)
        split_idx = int(len(all_track_ids) * 0.8)
        val_track_ids = set(all_track_ids[split_idx:])

    print("Building validation data...")
    valid_file = os.path.join(Config.save_dir, 'clip4cir_validation_cache.pkl')
    candidate_images, val_queries = pg.build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=Config.num_targets, cache_file=valid_file
    )
    if args.quick:
        candidate_images = candidate_images[:200]
        val_queries = val_queries[:20]
    print(f"Candidates: {len(candidate_images)}, Queries: {len(val_queries)}")

    val_dataset = pg.ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(Config.save_dir, f'candidate_feats_clip4cir.pt')
    )

    # ... 解析参数 ...
    if args.method == "coop":
        retriever = CoOpRetriever(clip_model, n_ctx=args.n_ctx).to(device)
    else:
        retriever = CoCoOpRetriever(clip_model, n_ctx=args.n_ctx).to(device)

    if args.weights:
        load_pretrained_weights(retriever, args.weights, device, args.method)
    else:
        print("⚠️ No pretrained weights provided. Using random initialization. Results will be poor.")

    print(f"\n=== Evaluating {args.method.upper()} ===")
    evaluate_retriever(retriever, val_dataset, device)

# ---------- 主训练函数 ----------
def train_cocoop():
    """使用自己的数据集训练 CoCoOp"""
    # 1. 划分训练/验证车辆
    split_file = os.path.join(Config.save_dir, 'track_split.pkl')
    if hasattr(pg, 'get_or_create_track_split'):
        train_track_ids, val_track_ids = pg.get_or_create_track_split(
            Config.track_ann_file, split_file, train_ratio=0.8, seed=42
        )
    else:
        random.seed(42)
        with open(Config.track_ann_file, 'r') as f:
            tracks = json.load(f)
        all_track_ids = list(tracks.keys())
        random.shuffle(all_track_ids)
        split_idx = int(len(all_track_ids) * 0.8)
        train_track_ids = set(all_track_ids[:split_idx])
        val_track_ids = set(all_track_ids[split_idx:])

    print(f"Train tracks: {len(train_track_ids)}, Val tracks: {len(val_track_ids)}")

    # 2. 构建训练集和验证集
    print("Building training triplets...")
    train_triplets = pg.build_train_triplets(
        Config.track_ann_file, Config.image_root, allowed_track_ids=train_track_ids
    )
    all_track_ids = sorted(set([t['track_id'] for t in train_triplets]))  # 排序保证确定性
    track_to_int = {tid: idx for idx, tid in enumerate(all_track_ids)}
    print(f"Training triplets: {len(train_triplets)}")

    # 使用互斥采样器（确保 batch 内车辆不重复）
    # train_dataset = pg.TripletDataset(train_triplets, preprocess)
    # sampler = pg.TrackMutualSampler(train_triplets, batch_size=Config.batch_size, shuffle=True)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=sampler,
    #     num_workers=Config.num_workers,
    #     pin_memory=True
    # )
    # 放弃批内互斥采样，使用普通随机采样
    train_dataset = pg.TripletDataset(train_triplets, preprocess, track_to_int)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print("Building validation data...")
    valid_file = os.path.join(Config.save_dir, 'clip4cir_validation_cache.pkl')
    candidate_images, val_queries = pg.build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=3, cache_file=valid_file
    )
    val_dataset = pg.ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(Config.save_dir, 'candidate_feats_clip4cir.pt')
    )
    print(f"Candidates: {len(candidate_images)}, Queries: {len(val_queries)}")

    # 3. 初始化模型
    retriever = CoCoOpRetriever(clip_model, n_ctx=4).to(device)
    optimizer = torch.optim.Adam(retriever.parameters(), lr=Config.lr)

    # 4. 训练循环
    best_map = 0.0
    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}")
        train_loss = train_epoch_cocoop(
            clip_model, retriever, train_loader, optimizer, device, Config.temperature
        )
        torch.save(retriever.state_dict(), os.path.join(Config.save_dir, 'train_temp_cocoop_retriever.pth'))
        print(f"Train Loss: {train_loss:.4f}")

        # 每 5 个 epoch 验证一次
        if epoch % 5 == 0:
            recalls, mAP = evaluate_retriever(retriever, val_dataset, device, Config.temperature)
            if mAP > best_map:
                best_map = mAP
                torch.save(retriever.state_dict(), os.path.join(Config.save_dir, 'best_cocoop_retriever.pth'))
                print("Best model saved.")

    print(f"Training finished. Best mAP: {best_map:.2f}%")


if __name__ == "__main__":
    train_cocoop()
    # import sys
    # # 直接在这里写参数，代替命令行
    # sys.argv = [
    #     "coop_cocoop_baseline.py",
    #     "--method", "cocoop",
    #     "--n_ctx", "4",
    #     "--weights", "./checkpoints/best_cocoop_retriever.pth"
    # ]
    # use_pre_train_model_val()