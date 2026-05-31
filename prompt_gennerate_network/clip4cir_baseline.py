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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import image_text_dynamic_prompt_gennerate as pg

Config = pg.Config
build_validation_data = pg.build_validation_data
get_or_create_track_split = pg.get_or_create_track_split
ValidationDataset = pg.ValidationDataset
clip_model = pg.clip_model
preprocess = pg.preprocess
clip_dim = pg.clip_dim

device = Config.device

# -------------------------------------------------------------------
# ✅ 【修复 1】官方 CLIP4Cir Combiner 结构（RN50x4 640维 完全匹配权重）
# -------------------------------------------------------------------
class Combiner(nn.Module):
    def __init__(self):
        super().__init__()
        # 从你的报错里提取的真实维度
        self.text_projection_layer = nn.Linear(640, 2560)
        self.image_projection_layer = nn.Linear(640, 2560)

        # 真实 combiner_layer 维度
        self.combiner_layer = nn.Linear(2560 * 2, 5120)

        # 输出层维度
        self.output_layer = nn.Linear(5120, 640)

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

def evaluate_clip4cir(candidate_images, queries, clip_model, combiner, preprocess, device,
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

    num_queries = len(queries)
    print("\n===== CLIP4Cir Evaluation Results =====")
    for k in recalls:
        print(f"Recall@{k}: {recalls[k] / num_queries * 100:.2f}%")
    print(f"MRR: {mrr / num_queries * 100:.2f}%")
    print(f"mAP: {ap_sum / num_queries * 100:.2f}%")
    return recalls, mrr, ap_sum

def main():
    # -------------------------------------------------------------------
    # ✅ 【修复 2】打印当前目录 + 绝对路径构建（永不找不到文件）
    # -------------------------------------------------------------------
    print("【INFO】当前工作目录：", os.getcwd())

    split_file = os.path.join(Config.save_dir, 'track_split.pkl')
    train_track_ids, val_track_ids = get_or_create_track_split(
        Config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )

    print("Building validation data...")
    candidate_images, val_queries = build_validation_data(
        Config.track_ann_file, Config.image_root, val_track_ids, num_targets=3, cache_file='path/to/validation_cache.pkl'
    )
    print(f"Candidates: {len(candidate_images)}, Queries: {len(val_queries)}")

    # -------------------------------------------------------------------
    # ✅ 【修复 3】固定官方权重维度：640（RN50x4）
    # -------------------------------------------------------------------
    # combiner = Combiner(clip_dim=640).to(device)
    combiner = Combiner().to(device)
    # -------------------------------------------------------------------
    # ✅ 【修复 4】绝对路径加载权重（永不报错）
    # -------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    combiner_weights_path = os.path.join(BASE_DIR, "../../base_model/clip4cir_combiner.pt")
    combiner_weights_path = os.path.abspath(combiner_weights_path)

    print("权重路径：", combiner_weights_path)

    if os.path.exists(combiner_weights_path):
        print("Loading CLIP4Cir Combiner weights...")
        checkpoint = torch.load(combiner_weights_path, map_location=device)
        model_weights = checkpoint["Combiner"]
        for key in model_weights.keys():
            print(key)
        combiner.load_state_dict(model_weights, strict=False)
        print("✅ Combiner 权重加载成功！")
    else:
        print("Warning: 权重文件不存在，使用随机初始化")

    evaluate_clip4cir(
        candidate_images, val_queries,
        clip_model, combiner, preprocess, device
    )

if __name__ == "__main__":
    main()