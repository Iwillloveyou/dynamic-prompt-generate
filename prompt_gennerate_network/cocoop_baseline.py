#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CoOp / CoCoOp 对比实验脚本（修复版）
真正使用可学习提示或实例条件提示生成查询特征
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

# ---------- CoOp 实现 ----------
class CoOpTextEncoder(nn.Module):
    """CoOp: 静态可学习上下文向量，与原始文本拼接"""
    def __init__(self, clip_model, n_ctx=4, ctx_init="", dtype=torch.float32):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = dtype
        self.n_ctx = n_ctx
        token_embedding = clip_model.token_embedding
        self.token_dim = token_embedding.embedding_dim

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            ctx_words = ctx_init.split(" ")
            n_ctx = len(ctx_words)
            with torch.no_grad():
                ctx_ids = clip.tokenize(ctx_init).to(device).squeeze(0)[1:-1]
                ctx_embeddings = token_embedding(ctx_ids).to(dtype)
            self.ctx = nn.Parameter(ctx_embeddings)
        else:
            n_ctx = self.n_ctx   # 保留原传入值
            self.ctx = nn.Parameter(torch.empty(n_ctx, self.token_dim, dtype=dtype))
            nn.init.normal_(self.ctx, std=0.02)
        # 关键：将实际使用的 n_ctx 赋值给 self.n_ctx
        self.n_ctx = n_ctx

    def forward(self, text_tokens):
        """
        text_tokens: [B, L] token ids
        返回增强后的文本特征 [B, D]
        """
        # token embeddings
        x = self.clip_model.token_embedding(text_tokens).type(self.dtype)  # [B, L, D]

        # 上下文向量扩展到 batch 维度
        ctx = self.ctx.unsqueeze(0).expand(text_tokens.size(0), -1, -1)  # [B, n_ctx, D]

        # 拼接：上下文向量放在文本 tokens 之前（也可之后，此处按常见做法）
        x = torch.cat([ctx, x], dim=1)  # [B, n_ctx+L, D]

        # 通过 CLIP 文本 transformer
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D]
        x = self.clip_model.ln_final(x).type(self.dtype)

        # 取 EOS token 位置的特征（原始 EOS 位置需要加上偏移 n_ctx）
        eos_positions = text_tokens.argmax(dim=-1) + self.n_ctx
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

        # 元网络
        self.meta_net = MetaNet(clip_dim, n_ctx, self.token_dim)

        # 可学习的基向量
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            ctx_words = ctx_init.split(" ")
            n_ctx = len(ctx_words)
            with torch.no_grad():
                ctx_ids = clip.tokenize(ctx_init).to(device).squeeze(0)[1:-1]
                ctx_embeddings = token_embedding(ctx_ids).to(self.dtype)
            self.ctx = nn.Parameter(ctx_embeddings)
        else:
            n_ctx = self.n_ctx   # 保留原传入值
            self.ctx = nn.Parameter(torch.empty(n_ctx, self.token_dim, dtype=self.dtype))
            nn.init.normal_(self.ctx, std=0.02)
        # 关键：将实际使用的 n_ctx 赋值给 self.n_ctx
        self.n_ctx = n_ctx

    def forward(self, text_tokens, img_feat):
        # 获取文本 token embeddings
        x = self.clip_model.token_embedding(text_tokens).type(self.dtype)  # [B, L, D]

        # 生成条件向量并与基向量相加
        ctx_cond = self.meta_net(img_feat)          # [B, n_ctx, D]
        ctx = ctx_cond + self.ctx_base.unsqueeze(0)  # [B, n_ctx, D]

        # 拼接
        x = torch.cat([ctx, x], dim=1)              # [B, n_ctx+L, D]

        # Transformer
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)

        # 取 EOS token
        eos_positions = text_tokens.argmax(dim=-1) + self.n_ctx
        x = x[torch.arange(x.size(0)), eos_positions] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1)


class CoCoOpRetriever(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init=""):
        super().__init__()
        self.text_encoder = CoCoOpTextEncoder(clip_model, n_ctx, ctx_init)

    def forward(self, text_tokens, img_feat):
        return self.text_encoder(text_tokens, img_feat)


# ---------- 评估函数（正确使用 retriever）----------
@torch.no_grad()
def evaluate_retriever(retriever, val_dataset, device, temperature=0.07, batch_size=64):
    retriever.eval()
    clip_model.eval()

    # 候选图像特征
    candidate_feats = val_dataset.load_or_extract_candidate_features(clip_model, device).to(device)

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
            sim_i = sim[i]
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
    print(f"mAP: {ap_sum / num_q * 100:.2f}%")
    return recalls, ap_sum


# ---------- 主函数 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="coop", choices=["coop", "cocoop"])
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
    candidate_images, val_queries = pg.build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=Config.num_targets
    )
    if args.quick:
        candidate_images = candidate_images[:200]
        val_queries = val_queries[:20]
    print(f"Candidates: {len(candidate_images)}, Queries: {len(val_queries)}")

    val_dataset = pg.ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(Config.save_dir, f'candidate_feats_{args.method}.pt')
    )

    # 创建检索器
    if args.method == "coop":
        retriever = CoOpRetriever(clip_model, n_ctx=args.n_ctx).to(device)
    else:
        retriever = CoCoOpRetriever(clip_model, n_ctx=args.n_ctx).to(device)

    print(f"\n=== Evaluating {args.method.upper()} ===")
    evaluate_retriever(retriever, val_dataset, device)


if __name__ == "__main__":
    main()