import os
import json
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.utils.data import Dataset, DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import image_text_dynamic_prompt_gennerate as pg

Config = pg.Config
build_validation_data = pg.build_validation_data
get_or_create_track_split = pg.get_or_create_track_split
ValidationDataset = pg.ValidationDataset
TripletDataset=pg.TripletDataset
TrackMutualSampler=pg.TrackMutualSampler
clip_model = pg.clip_model
preprocess = pg.preprocess
clip_dim = pg.clip_dim

device = Config.device

# -------------------------------------------------------------------
# ✅ 【修复 1】官方 CLIP4Cir Combiner 结构（RN50x4 640维 完全匹配权重）
# -------------------------------------------------------------------
class Combiner(nn.Module):
    def __init__(self, input_dim=640):
        super().__init__()
        # 从你的报错里提取的真实维度 跟自己的特征维度一致，自定义是512，已有预训练模型是644，使用时需用adapter转换
        self.text_projection_layer = nn.Linear(input_dim, 2560)
        self.image_projection_layer = nn.Linear(input_dim, 2560)

        # 真实 combiner_layer 维度
        self.combiner_layer = nn.Linear(2560 * 2, 5120)

        # 输出层维度
        self.output_layer = nn.Linear(5120, input_dim)

        # 动态标量层（你的权重是 4 层：0,1,2,3）
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(5120, 5120),
            nn.ReLU(),
            nn.Linear(5120, 5120),  # 你权重里有这一层！
            nn.ReLU(),
            nn.Linear(5120, 1),
            nn.Sigmoid()
        )

    def forward(self, img_feat, txt_feat):
        img = self.image_projection_layer(img_feat)
        txt = self.text_projection_layer(txt_feat)

        # 拼接
        combined = torch.cat([img, txt], dim=-1)
        combined = F.relu(self.combiner_layer(combined))

        s = self.dynamic_scalar(combined)
        out = self.output_layer(combined)
        return F.normalize(out, dim=-1)

def train_epoch_clip4cir(clip_model, combiner, train_loader, optimizer, device, temperature=0.07):
    """训练一个epoch"""
    clip_model.eval()
    combiner.train()
    total_loss = 0
    num_batches = 0
    # 使用预训练权重时打开
    # adapter = nn.Linear(512, 640).to(device)

    for ref_imgs, target_imgs, texts, track_ids in tqdm(train_loader, desc="Training"):
        ref_imgs = ref_imgs.to(device)
        target_imgs = target_imgs.to(device)
        texts = texts.to(device)
        batch_size = ref_imgs.size(0)

        with torch.no_grad():
            ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1).float()
            target_feat = F.normalize(clip_model.encode_image(target_imgs), dim=-1).float()
            text_feat = F.normalize(clip_model.encode_text(texts), dim=-1).float()

        # Combiner 融合
        # ref_feat = adapter(ref_feat)      # [batch, 640]
        # text_feat = adapter(text_feat)
        # target_feat = adapter(target_feat)
        query_feat = combiner(ref_feat, text_feat)

        # 对比损失
        # logits = query_feat @ target_feat.T / temperature
        # labels = torch.arange(batch_size, device=device)
        # loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        # 使用多正样本对比损失
        track_ids = track_ids.to(device)   # 移动标签到 GPU
        loss = multi_positive_contrastive_loss(query_feat, target_feat, track_ids, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

@torch.no_grad()
def evaluate_clip4cir(combiner, val_dataset, device, temperature=0.07, batch_size=64):
    """评估CLIP4Cir模型"""
    combiner.eval()
    clip_model.eval()

    candidate_feats = val_dataset.load_or_extract_candidate_features(clip_model, device).to(device).float()
    queries = val_dataset.queries
    num_queries = len(queries)

    # 由于使用和s原网络一样的ViT-B/32model，输出是512，但模型是640，所以进行转换。或者加载 RN50x4 模型，在开头使用model, preprocess = clip.load("RN50x4", device=device)
    # adapter = nn.Linear(512, 640).to(device)

    # 先确保 candidate_feats 是 [num_candidates, 512] 格式
    if candidate_feats.shape[0] == 512:  # 如果第一维是特征维度
        candidate_feats = candidate_feats.T  # 转置为 [num_candidates, 512]
    # 可选：从预训练权重的 image_projection_layer 中提取前 512 列作为初始化（略）
    # candidate_feats = adapter(candidate_feats)  # [num_candidates, 640]

    recalls = {1: 0, 5: 0, 10: 0}
    ap_sum = 0.0
    ndcg_sum = {5: 0.0, 10: 0.0}   # 新增 NDCG@5, NDCG@10
    # adapter = nn.Linear(512, 640).to(device)

    for start in tqdm(range(0, num_queries, batch_size), desc="Evaluating"):
        end = min(start + batch_size, num_queries)
        batch_queries = queries[start:end]

        ref_imgs = []
        texts = []
        target_idxss = []

        for q in batch_queries:
            ref_img = Image.open(q['ref_img']).convert('RGB')
            ref_tensor = preprocess(ref_img).unsqueeze(0).to(device)
            ref_imgs.append(ref_tensor)
            text_tokens = clip.tokenize(q['caption']).to(device)
            texts.append(text_tokens)
            target_idxss.append(q['target_idxs'])

        ref_imgs = torch.cat(ref_imgs, dim=0)
        texts = torch.cat(texts, dim=0)

        ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1).float()
        text_feat = F.normalize(clip_model.encode_text(texts), dim=-1).float()
        # ref_feat = adapter(ref_feat)      # [batch, 640]
        # text_feat = adapter(text_feat)
        query_feat = combiner(ref_feat, text_feat)

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

            # AP 计算
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

            # Recall@K
            first_rank = None
            for rank, idx in enumerate(sorted_indices.cpu().tolist()):
                if idx in pos_idxs:
                    first_rank = rank
                    break
            if first_rank is not None:
                for k in recalls:
                    if first_rank < k:
                        recalls[k] += 1

            # ---- 新增 NDCG 计算 ----
            # 构建相关性字典
            is_relevant = {idx: 1 for idx in pos_idxs}
            # 获取前 K 个结果的相关性得分列表
            for k in [5, 10]:
                dcg = 0.0
                for rank, idx in enumerate(sorted_indices[:k]):
                    gain = is_relevant.get(idx.item(), 0)
                    dcg += gain / np.log2(rank + 2)  # rank从0开始，分母log2(rank+2)
                # 理想 DCG（所有正样本排在最前）
                ideal_gains = [1] * min(P, k)
                idcg = sum(g / np.log2(i+2) for i, g in enumerate(ideal_gains))
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_sum[k] += ndcg

    num_q = len(queries)
    print("\n===== CLIP4Cir Evaluation =====")
    for k in recalls:
        print(f"Recall@{k}: {recalls[k] / num_q * 100:.2f}%")
    print(f"mAP: {ap_sum / num_q * 100:.2f}%")
    print(f"NDCG@5: {ndcg_sum[5]:.2f}%, NDCG@10: {ndcg_sum[10]:.2f}%")
    return recalls, ap_sum / num_q * 100, ndcg_sum


def multi_positive_contrastive_loss(query_feat, target_feat, track_ids, temperature=0.07):
    device = query_feat.device
    # 确保 track_ids 在正确设备上且为整数
    if not isinstance(track_ids, torch.Tensor):
        track_ids = torch.tensor(track_ids, device=device)
    else:
        track_ids = track_ids.to(device)
    track_ids = track_ids.long().view(-1)

    sim = query_feat @ target_feat.T / temperature
    pos_mask = (track_ids[:, None] == track_ids[None, :]).float()
    # 可选：去除对角线（避免自身匹配）
    # pos_mask = pos_mask * (1 - torch.eye(track_ids.size(0), device=device))
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)
    loss = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
    return loss

def evaluate_clip4cir_by_pre_train_model(candidate_images, queries, clip_model, combiner, preprocess, device,
                      temperature=0.07, batch_size=64):
    clip_model.eval()
    combiner.eval()

    print("Extracting/loading candidate features...")
    tmp_dataset = ValidationDataset(
        candidate_images, [], preprocess,
        cache_path=os.path.join(Config.save_dir, 'candidate_feats_clip4cir.pt')
    )
    candidate_feats = tmp_dataset.load_or_extract_candidate_features(clip_model, device)
    candidate_feats = candidate_feats.to(device).float()

    # 由于使用和s原网络一样的ViT-B/32model，输出是512，但模型是640，所以进行转换。或者加载 RN50x4 模型，在开头使用model, preprocess = clip.load("RN50x4", device=device)
    adapter = nn.Linear(512, 640).to(device)

    # 先确保 candidate_feats 是 [num_candidates, 512] 格式
    if candidate_feats.shape[0] == 512:  # 如果第一维是特征维度
        candidate_feats = candidate_feats.T  # 转置为 [num_candidates, 512]
    # 可选：从预训练权重的 image_projection_layer 中提取前 512 列作为初始化（略）
    candidate_feats = adapter(candidate_feats)  # [num_candidates, 640]

    num_queries = len(queries)
    recalls = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    ap_sum = 0.0
    ndcg_sum = {5: 0.0, 10: 0.0}   # 新增 NDCG@5, NDCG@10

    for start in tqdm(range(0, num_queries, batch_size), desc="Evaluating queries (CLIP4Cir)"):
        end = min(start + batch_size, num_queries)
        batch_queries = queries[start:end]

        batch_ref_imgs = []
        batch_texts = []
        batch_target_idxs = []

        for q in batch_queries:
            ref_img = Image.open(q['ref_img']).convert('RGB')
            ref_tensor = preprocess(ref_img).unsqueeze(0)
            batch_ref_imgs.append(ref_tensor)

            text_tokens = clip.tokenize(q['caption']).squeeze(0)
            batch_texts.append(text_tokens)
            batch_target_idxs.append(q['target_idxs'])

        batch_ref_imgs = torch.cat(batch_ref_imgs, dim=0).to(device)
        batch_texts = torch.stack(batch_texts, dim=0).to(device)

        with torch.no_grad():
            ref_feat = clip_model.encode_image(batch_ref_imgs)
            ref_feat = F.normalize(ref_feat, dim=-1).float()
            text_feat = clip_model.encode_text(batch_texts)
            text_feat = F.normalize(text_feat, dim=-1).float()
            ref_feat = adapter(ref_feat)      # [batch, 640]
            text_feat = adapter(text_feat)
            query_feat = combiner(ref_feat, text_feat)

        sim = query_feat @ candidate_feats.T

        for i in range(len(batch_queries)):
            sim_i = sim[i]
            sorted_indices = sim_i.argsort(descending=True)
            pos_idxs = batch_target_idxs[i]
            P = len(pos_idxs)

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

            first_rank = None
            for rank, idx in enumerate(sorted_indices.cpu().tolist()):
                if idx in pos_idxs:
                    first_rank = rank
                    break
            if first_rank is not None:
                for k in recalls:
                    if first_rank < k:
                        recalls[k] += 1
                mrr += 1.0 / (first_rank + 1)

            # ---- 新增 NDCG 计算 ----
            # 构建相关性字典
            is_relevant = {idx: 1 for idx in pos_idxs}
            # 获取前 K 个结果的相关性得分列表
            for k in [5, 10]:
                dcg = 0.0
                for rank, idx in enumerate(sorted_indices[:k]):
                    gain = is_relevant.get(idx.item(), 0)
                    dcg += gain / np.log2(rank + 2)  # rank从0开始，分母log2(rank+2)
                # 理想 DCG（所有正样本排在最前）
                ideal_gains = [1] * min(P, k)
                idcg = sum(g / np.log2(i+2) for i, g in enumerate(ideal_gains))
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_sum[k] += ndcg

    num_queries = len(queries)
    print("\n===== CLIP4Cir Evaluation Results =====")
    for k in recalls:
        print(f"Recall@{k}: {recalls[k] / num_queries * 100:.2f}%")
    print(f"MRR: {mrr / num_queries * 100:.2f}%")
    print(f"mAP: {ap_sum / num_queries * 100:.2f}%")
    print(f"NDCG@5: {ndcg_sum[5]:.2f}%, NDCG@10: {ndcg_sum[10]:.2f}%")
    return recalls, mrr, ap_sum, ndcg_sum


def define_train():
    # 1. 划分训练/验证车辆
    split_file = os.path.join(Config.save_dir, 'track_split.pkl')
    train_track_ids, val_track_ids = get_or_create_track_split(
        Config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )
    print(f"Train tracks: {len(train_track_ids)}, Val tracks: {len(val_track_ids)}")

    # 2. 构建训练集和验证集
    print("Building training triplets...")
    train_triplets = pg.build_train_triplets(
        Config.track_ann_file, Config.image_root, allowed_track_ids=train_track_ids
    )
    all_track_ids = sorted(set([t['track_id'] for t in train_triplets]))  # 排序保证确定性
    track_to_int = {tid: idx for idx, tid in enumerate(all_track_ids)}
    print(f"Training triplets: {len(train_triplets)}")

    # 使用互斥采样器（确保batch内车辆不重复）
    # train_dataset = pg.TripletDataset(train_triplets, preprocess)
    # sampler = TrackMutualSampler(train_triplets, batch_size=Config.batch_size, shuffle=True)
    # train_loader = DataLoader(
    #     train_dataset, batch_sampler=sampler,
    #     num_workers=Config.num_workers, pin_memory=True
    # )
    # 创建 Dataset
    train_dataset = TripletDataset(train_triplets, preprocess, track_to_int)

    # 使用普通的 DataLoader（shuffle=True，不自定义 sampler）
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
    candidate_images, val_queries = build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=3, cache_file=valid_file
    )
    val_dataset = ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(Config.save_dir, 'candidate_feats_clip4cir.pt')
    )
    print(f"Validation: {len(candidate_images)} candidates, {len(val_queries)} queries")

    # 3. 初始化模型
    combiner = Combiner(input_dim=512).to(device)
    optimizer = torch.optim.Adam(combiner.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 50个epoch

    # 4. 训练循环
    best_map = 0.0
    patience = 5
    early_stop_count = 0
    # 断点文件路径
    ckpt_path = os.path.join(Config.save_dir, "resume_checkpoint.pth")
    start_epoch = 1

    # ========= 加载断点：存在就恢复所有训练状态 =========
    if os.path.exists(ckpt_path):
        # 解决你之前报错weights_only，加weights_only=False
        checkpoint = torch.load(ckpt_path, map_location=Config.device)
        combiner.load_state_dict(checkpoint["combiner"])
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["sch"])
        start_epoch = checkpoint["epoch"] + 1  # 中断epoch跑完了，下一轮+1开始
        best_map = checkpoint["best_map"]
        early_stop_count = checkpoint["stop_cnt"]
        print(f"✅ 断点加载成功，从Epoch {start_epoch} 继续训练，历史best_mAP={best_map:.4f}")

    for epoch in range(start_epoch, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}")
        train_loss = train_epoch_clip4cir(
            clip_model, combiner, train_loader, optimizer, device, Config.temperature
        )
        torch.save(combiner.state_dict(), os.path.join(Config.save_dir, 'train_temp_clip4cir_refine_combiner.pth'))
        print(f"Train Loss: {train_loss:.4f}")

        # 每2个epoch验证一次
        if epoch % 2 == 0:
            recalls, mAP, ndcg_sum = evaluate_clip4cir(combiner, val_dataset, device, Config.temperature)
            # mrr是什么指标，如果要计算map，该怎么修改
            if mAP > best_map:
                best_map = mAP
                early_stop_count = 0
                torch.save(combiner.state_dict(), os.path.join(Config.save_dir, 'best_clip4cir_refine_combiner.pth'))
                print("New Best model saved!")
            else:
                early_stop_count += 1
                print(f"No improve, early_stop count: {early_stop_count}/{patience}")

        # ========= 每轮训练完保存完整断点（断电/手动中断都能续） =========
        save_dict = {
            "combiner": combiner.state_dict(),
            "opt": optimizer.state_dict(),
            "sch": scheduler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "stop_cnt": early_stop_count
        }
        torch.save(save_dict, ckpt_path)

        # 早停判断
        if early_stop_count >= patience:
            print(f"Early Stop Trigger! {patience} epochs no mAP improve, exit training.")
            break

        scheduler.step()
    print("Training finished.")

def use_pre_model_val():
    # -------------------------------------------------------------------
    # ✅ 【修复 2】打印当前目录 + 绝对路径构建（永不找不到文件）
    # -------------------------------------------------------------------
    print("【INFO】当前工作目录：", os.getcwd())

    split_file = os.path.join(Config.save_dir, 'track_split.pkl')
    train_track_ids, val_track_ids = get_or_create_track_split(
        Config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )

    print("Building validation data...")
    valid_file = os.path.join(Config.save_dir, 'clip4cir_validation_cache.pkl')
    candidate_images, val_queries = build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=3, cache_file=valid_file
    )
    val_dataset = ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(Config.save_dir, 'candidate_feats_clip4cir.pt')
    )
    print(f"Candidates: {len(candidate_images)}, Queries: {len(val_queries)}")

    # -------------------------------------------------------------------
    # ✅ 【修复 3】固定官方权重维度：640（RN50x4）
    # -------------------------------------------------------------------
    # combiner = Combiner(clip_dim=640).to(device)
    combiner = Combiner(input_dim=512).to(device)
    # -------------------------------------------------------------------
    # ✅ 【修复 4】绝对路径加载权重（永不报错）
    # -------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    combiner_weights_path = os.path.join(Config.save_dir, 'train_temp_clip4cir_refine_combiner.pth')
    combiner_weights_path = os.path.abspath(combiner_weights_path)

    print("权重路径：", combiner_weights_path)

    if os.path.exists(combiner_weights_path):
        print("Loading CLIP4Cir Combiner weights...")
        checkpoint = torch.load(combiner_weights_path, map_location=device)
        # 1. 提取模型参数字典（尝试常见 key）
        if "Combiner" in checkpoint:
            model_weights = checkpoint["Combiner"]
        if "state_dict" in checkpoint:
            model_weights = checkpoint["state_dict"]
        elif "model" in checkpoint:
            model_weights = checkpoint["model"]
        else:
            model_weights = checkpoint  # 直接就是参数字典
        for key in model_weights.keys():
            print(key)
        combiner.load_state_dict(model_weights, strict=False)
        print("✅ Combiner 权重加载成功！")
    else:
        print("Warning: 权重文件不存在，使用随机初始化")

    evaluate_clip4cir(combiner, val_dataset, device, Config.temperature)
    #  evaluate_clip4cir_by_pre_train_model(
    #     candidate_images, val_queries,
    #     clip_model, combiner, preprocess, device
    # )

if __name__ == "__main__":
    define_train()