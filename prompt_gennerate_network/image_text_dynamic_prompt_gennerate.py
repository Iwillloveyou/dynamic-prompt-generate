import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import Sampler

# -------------------- 配置 --------------------
class Config:
    # 数据路径（请根据实际情况修改）
    data_root = './cityflow-nl-data/'            # 原始数据根目录（包含 train-tracks.json 等）
    image_root = './cityflow-nl-data/train/'     # 提取的图像根目录（包含 S01, S03 等）
    track_ann_file = os.path.join(data_root, 'train-tracks.json')   # 车辆轨迹标注
    # CLIP 模型
    clip_model_name = 'ViT-B/32'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 概念向量库
    concept_vector_path = './concept_vectors.npy'
    # 训练参数
    batch_size = 32
    epochs = 50
    lr = 1e-4
    temperature = 0.07
    hidden_dim = 512
    num_workers = 4
    # 保存路径
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    # 验证时候选集大小（-1 表示使用全部候选）
    val_candidate_size = -1

config = Config()

# -------------------- 加载 CLIP 和概念向量 --------------------
print("Loading CLIP model...")
clip_model, preprocess = clip.load(config.clip_model_name, device=config.device)
for param in clip_model.parameters():
    param.requires_grad = False
clip_dim = clip_model.visual.output_dim

print("Loading concept vectors...")
concept_vectors = np.load(config.concept_vector_path)
concept_vectors = torch.from_numpy(concept_vectors).float().to(config.device)
concept_vectors = F.normalize(concept_vectors, dim=-1)
num_concepts = concept_vectors.size(0)

def compute_prior_scores(query_feat, concept_extend_embs):
    """
    query_feat: [B, D]
    concept_extend_embs: list of list of tensors (each shape [D])
    returns: [B, C] prior scores
    """
    B = query_feat.size(0)
    C = len(concept_extend_embs)
    scores = torch.zeros(B, C, device=query_feat.device)
    for c, exts in enumerate(concept_extend_embs):
        if len(exts) == 0:
            continue
        ext_mat = torch.stack(exts, dim=0)  # [M, D]
        sim = query_feat @ ext_mat.T        # [B, M]
        scores[:, c] = sim.max(dim=1)[0]    # 最大相似度
    return scores

# -------------------- PromptGenerator --------------------
class PromptGenerator(nn.Module):
    def __init__(self, concept_name_embs, concept_extend_embs, clip_dim, num_concepts, hidden_dim=256):
        super().__init__()
        self.concept_names = concept_name_embs   # [C, D]
        self.concept_extend_embs = concept_extend_embs  # list of list

        # 可学习 MLP：输入是 先验得分 + 原始查询特征（可选）
        self.mlp = nn.Sequential(
            nn.Linear(num_concepts + clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, text_feat, img_feat=None):
        # 1. 融合文本和图像得到查询特征
        if img_feat is not None:
            query_feat = F.normalize(text_feat + img_feat, dim=-1)
        else:
            query_feat = text_feat

        # 2. 计算先验得分（基于扩展描述的最大相似度）
        prior = compute_prior_scores(query_feat, self.concept_extend_embs)   # [B, C]

        # 3. 将先验得分与原始查询特征拼接，输入 MLP 得到调整后的 logits
        mlp_input = torch.cat([prior, query_feat], dim=-1)   # [B, C + D]
        logits = self.mlp(mlp_input)                         # [B, C]

        # 4. 应用温度和 softmax
        weights = F.softmax(logits / self.temperature, dim=-1)

        # 5. 加权组合概念名称向量生成提示
        dyn_prompt = weights @ self.concept_names            # [B, D]

        # 6. 融合回原始文本特征
        combined = text_feat + self.alpha * dyn_prompt
        combined = F.normalize(combined, dim=-1)
        return combined

# -------------------- SemanticPromptGenerator --------------------
class SemanticPromptGenerator(nn.Module):
    def __init__(self, concept_name_embs, concept_extend_embs, temperature=0.07):
        """
        concept_name_embs: [C, D] 概念名称 embedding (已归一化)
        concept_extend_embs: list of list of [D] tensors (每个概念的扩展描述 embedding，已归一化)
        """
        super().__init__()
        self.register_buffer('concept_names', concept_name_embs)   # [C, D]
        # 存储扩展描述 embeddings，每个概念的扩展可能长度不一，需要用 list 保存（不能直接注册 buffer）
        self.concept_extend_embs = concept_extend_embs   # list of list of tensors
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, text_feat, img_feat=None):
        """
        text_feat: [B, D]
        img_feat: [B, D] or None
        """
        if img_feat is not None:
            # 融合两种模态：简单相加或拼接投影？这里取两者平均（可以调整）
            query_feat = F.normalize(text_feat + img_feat, dim=-1)
        else:
            query_feat = text_feat

        batch_size = query_feat.size(0)
        C = self.concept_names.size(0)

        # 计算每个查询与每个概念的扩展描述的相似度，取最大值
        concept_scores = torch.zeros(batch_size, C, device=query_feat.device)
        for c in range(C):
            # 该概念的所有扩展描述 embedding
            exts = self.concept_extend_embs[c]   # list of tensors [D]
            if len(exts) == 0:
                # 如果没有扩展描述，则使用概念名称本身
                exts = [self.concept_names[c]]
            # 将所有扩展描述堆叠为 [M, D]
            exts_tensor = torch.stack(exts, dim=0).to(query_feat.device)  # [M, D]
            # 计算相似度矩阵 [B, M]
            sim = query_feat @ exts_tensor.T
            # 取每个查询的最大值作为该概念的分值
            concept_scores[:, c] = sim.max(dim=1)[0]

        # 应用温度系数
        logits = concept_scores / self.temperature
        weights = F.softmax(logits, dim=-1)   # [B, C]

        # 加权组合概念名称向量
        dyn_prompt = weights @ self.concept_names   # [B, D]

        # 融合原始文本特征与动态提示
        combined = text_feat + dyn_prompt
        combined = F.normalize(combined, dim=-1)
        return combined

# -------------------- 数据集构建 --------------------
def build_train_triplets(track_ann_file, image_root, allowed_track_ids=None):
    """
    从 train-tracks.json allowed_track_ids构建训练三元组列表
    返回: list of dict, 每个包含 ref_img_path, target_img_path, caption
    """
    with open(track_ann_file, 'r') as f:
        tracks = json.load(f)
    triplets = []
    for track_id, info in tracks.items():
        if allowed_track_ids is not None and track_id not in allowed_track_ids:
            continue
        frames = info['frames']
        captions = info['nl']
        if len(frames) < 2 or len(captions) == 0:
            continue
        for cap in captions:
            ref_img_path, target_img_path = random.sample(frames, 2)
            # 构建绝对路径
            ref_full = os.path.join(image_root, ref_img_path.lstrip('./'))
            target_full = os.path.join(image_root, target_img_path.lstrip('./'))
            triplets.append({
                'ref_img': ref_full,
                'target_img': target_full,
                'caption': cap,
                'track_id': track_id
            })
    return triplets

def build_validation_data(track_ann_file, image_root, val_track_ids, num_targets=2):
    """
    划分验证集车辆，构建：
        candidate_images: 所有验证车辆的全部图像路径列表
        queries: 每个查询包含 ref_img_path, caption, target_img_path (在 candidate_images 中的索引)
    """
    with open(track_ann_file, 'r') as f:
        tracks = json.load(f)
    candidate_images = []
    candidate_track_ids = []
    img_to_idx = {}
    for tid in val_track_ids:
        if tid not in tracks:
            continue
        frames = tracks[tid]['frames']
        for frame in frames:
            img_path = os.path.join(image_root, frame)
            candidate_images.append(img_path)
            candidate_track_ids.append(tid)
            img_to_idx[img_path] = len(candidate_images) - 1
    queries = []
    for tid in val_track_ids:
        frames = tracks[tid]['frames']
        captions = tracks[tid]['nl']
        if len(frames) < 2 or len(captions) == 0:
            continue
        # 为避免每次验证结果波动过大，固定采样策略：对每条描述，随机选择参考图像，再随机选择多个目标
        for cap in captions:
            # 随机选择参考图像（从所有图像中选一张）
            ref_frame = random.choice(frames)
            # 剩余图像作为候选目标池
            other_frames = [f for f in frames if f != ref_frame]
            if len(other_frames) == 0:
                continue
            # 确定要采样的目标数量
            n_targets = min(num_targets, len(other_frames))
            target_frames = random.sample(other_frames, n_targets) if n_targets > 0 else []
            if not target_frames:
                continue
            ref_full = os.path.join(image_root, ref_frame.lstrip('./'))
            target_idxs = []
            for target_img in target_frames:
                target_full = os.path.join(image_root, target_img.lstrip('./'))
                if target_full in img_to_idx:
                    target_idxs.append(img_to_idx[target_full])
            if not target_idxs:
                continue
            queries.append({
                'ref_img': ref_full,
                'caption': cap,
                'target_idxs': target_idxs,   # 存储多个索引
                'track_id': tid
            })
    return candidate_images, queries

# -------------------- 数据集类 --------------------
class TripletDataset(Dataset):
    def __init__(self, triplets, preprocess):
        self.triplets = triplets
        self.preprocess = preprocess

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        ref_img = Image.open(item['ref_img']).convert('RGB')
        target_img = Image.open(item['target_img']).convert('RGB')
        ref_img = self.preprocess(ref_img)
        target_img = self.preprocess(target_img)
        text = clip.tokenize(item['caption'], truncate=True).squeeze(0)
        return ref_img, target_img, text, item['track_id']   # 增加 track_id

class ValidationDataset(Dataset):
    """验证集：存储所有候选图像和所有查询"""
    def __init__(self, candidate_images, queries, preprocess, cache_path=None):
        self.candidate_images = candidate_images
        self.queries = queries
        self.preprocess = preprocess
        self.cache_path = cache_path
        self.candidate_feats = None   # 存储候选特征，延迟加载

    def load_or_extract_candidate_features(self, clip_model, device):
        """加载缓存或提取候选特征，并保持在内存中"""
        if self.candidate_feats is not None:
            return self.candidate_feats

        # 如果有缓存文件，直接加载
        if self.cache_path is not None and os.path.exists(self.cache_path):
            print(f"Loading cached candidate features from {self.cache_path}")
            self.candidate_feats = torch.load(self.cache_path)
            return self.candidate_feats

        # 否则提取特征
        print("Extracting candidate features...")
        feats = []
        for img_path in tqdm(self.candidate_images, desc="Encoding candidates"):
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(img_tensor)
                feat = F.normalize(feat, dim=-1).cpu()
            feats.append(feat.squeeze(0))
        self.candidate_feats = torch.stack(feats)   # [C, D]

        # 保存到缓存
        if self.cache_path is not None:
            torch.save(self.candidate_feats, self.cache_path)
            print(f"Cached candidate features to {self.cache_path}")
        return self.candidate_feats

    def __getitem__(self, idx):
        # 与之前相同，不涉及特征提取
        query = self.queries[idx]
        ref_img = Image.open(query['ref_img']).convert('RGB')
        ref_img = self.preprocess(ref_img)
        text = clip.tokenize(query['caption'], truncate=True).squeeze(0)
        return ref_img, text, query['target_idx'], query['track_id']

class TrackMutualSampler(Sampler):
    def __init__(self, triplets, batch_size, shuffle=True):
        # 按 track_id 分组
        self.track_to_indices = defaultdict(list)
        for idx, t in enumerate(triplets):
            self.track_to_indices[t['track_id']].append(idx)
        self.track_ids = list(self.track_to_indices.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # 每个 epoch 随机打乱 track 顺序
        track_ids = self.track_ids.copy()
        if self.shuffle:
            random.shuffle(track_ids)
        batch = []
        for tid in track_ids:
            # 从当前 track 的所有三元组中随机选择一个
            idx = random.choice(self.track_to_indices[tid])
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        # 丢弃最后不足 batch_size 的部分（或可补全，但丢弃更简单）
        if len(batch) > 0:
            # 可选：补充一些样本或直接丢弃
            pass

    def __len__(self):
        return len(self.track_ids) // self.batch_size

# -------------------- 训练函数 --------------------
def train_epoch(clip_model, generator, dataloader, optimizer, device, temperature):
    clip_model.eval()
    generator.train()
    total_loss = 0
    num_batches = 0
    for ref_imgs, target_imgs, texts in tqdm(dataloader, desc='Training'):
        ref_imgs = ref_imgs.to(device)
        target_imgs = target_imgs.to(device)
        texts = texts.to(device)
        batch_size = ref_imgs.size(0)

        with torch.no_grad():
            ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1)
            target_feat = F.normalize(clip_model.encode_image(target_imgs), dim=-1)
            text_feat = F.normalize(clip_model.encode_text(texts), dim=-1)

        query_feat = generator(text_feat, ref_feat)   # [B, D]
        logits = query_feat @ target_feat.T / temperature
        labels = torch.arange(batch_size, device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

# -------------------- 验证函数 --------------------
@torch.no_grad()
def evaluate(clip_model, generator, val_dataset, device, temperature, k_list=[1,5,10]):
    """
    在验证集上进行检索评估
    """
    clip_model.eval()
    generator.eval()

    # 预先提取所有候选图像特征
    candidate_feats = val_dataset.get_candidate_features(clip_model, device).to(device)  # [C, D]

    queries = val_dataset.queries
    num_queries = len(queries)
    recalls = {k: 0 for k in k_list}
    mrr = 0.0

    for idx in tqdm(range(num_queries), desc="Evaluating"):
        ref_img, text, target_idx, _ = val_dataset[idx]
        ref_img = ref_img.unsqueeze(0).to(device)
        text = text.unsqueeze(0).to(device)

        ref_feat = F.normalize(clip_model.encode_image(ref_img), dim=-1)
        text_feat = F.normalize(clip_model.encode_text(text), dim=-1)
        query_feat = generator(text_feat, ref_feat)   # [1, D]

        # 计算与所有候选图像的相似度
        sim = query_feat @ candidate_feats.T   # [1, C]
        sim = sim.squeeze(0) / temperature
        # 按降序排序，得到排序后的索引
        sorted_indices = sim.argsort(descending=True)
        # 找到目标图像在排序中的位置（从0开始）
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item()

        # 更新 Recall@K
        for k in k_list:
            if rank < k:
                recalls[k] += 1
        # 更新 MRR todo:mrr是做什么的 可以计算map吗
        mrr += 1.0 / (rank + 1)

    for k in k_list:
        recalls[k] = recalls[k] / num_queries * 100
    mrr = mrr / num_queries * 100
    return recalls, mrr

#批次计算验证 效率更高
@torch.no_grad()
def evaluate_batched(clip_model, generator, val_dataset, device, temperature, batch_size=64, k_list=[1,5,10]):
    clip_model.eval()
    generator.eval()

    candidate_feats = val_dataset.load_or_extract_candidate_features(clip_model, device).to(device)  # [C, D]
    queries = val_dataset.queries
    num_queries = len(queries)

    recalls = {k: 0 for k in k_list}
    ap_sum = 0.0   # 累积所有查询的 AP

    for start in tqdm(range(0, num_queries, batch_size), desc="Evaluating batches"):
        end = min(start + batch_size, num_queries)
        batch_queries = queries[start:end]
        batch_size_actual = end - start

        ref_imgs = []
        texts = []
        target_idxss = []   # 每个查询的正样本索引列表 (list of list)
        for q in batch_queries:
            ref_img = Image.open(q['ref_img']).convert('RGB')
            ref_img = val_dataset.preprocess(ref_img)
            ref_imgs.append(ref_img)
            texts.append(clip.tokenize(q['caption'], truncation=True).squeeze(0))
            target_idxss.append(q['target_idxs'])   # 存储列表

        ref_imgs = torch.stack(ref_imgs).to(device)
        texts = torch.stack(texts).to(device)

        ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1)
        text_feat = F.normalize(clip_model.encode_text(texts), dim=-1)
        query_feat = generator(text_feat, ref_feat)   # [B, D]

        sim = query_feat @ candidate_feats.T / temperature   # [B, C]

        for i in range(batch_size_actual):
            sim_i = sim[i]                      # [C]
            sorted_indices = sim_i.argsort(descending=True)
            pos_idxs = target_idxss[i]          # 该查询的所有正样本索引列表
            P = len(pos_idxs)  # 正样本总数

            # ---- 计算 AP ----
            # 构建相关标识数组
            is_relevant = torch.zeros(len(candidate_feats), dtype=torch.bool, device=device)
            for idx in pos_idxs:
                is_relevant[idx] = True
            sorted_relevant = is_relevant[sorted_indices]   # [C]
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

            # ---- 计算 Recall@K（只要有一个正样本命中前K）----
            # 找到第一个正样本的排名位置
            first_rank = None
            for rank_idx, idx in enumerate(sorted_indices.cpu().tolist()):
                if idx in pos_idxs:
                    first_rank = rank_idx
                    break
            if first_rank is not None:
                for k in k_list:
                    if first_rank < k:
                        recalls[k] += 1
    # 平均指标
    for k in k_list:
        recalls[k] = recalls[k] / num_queries * 100
    mAP = ap_sum / num_queries * 100
    return recalls, mAP

# -------------------- 推理示例 -------------------- candidate_image_paths是整个数据集的图像吗
def retrieve(query_text, query_image_path, candidate_image_paths, clip_model, generator, preprocess, device, temperature=0.07, top_k=5):
    """
    给定文本查询和参考图像，从候选图像列表中检索最相似的图像
    """
    # 加载并预处理查询图像
    ref_img = Image.open(query_image_path).convert('RGB')
    ref_img_tensor = preprocess(ref_img).unsqueeze(0).to(device)
    # 预处理候选图像
    candidate_tensors = []
    for path in candidate_image_paths:
        img = Image.open(path).convert('RGB')
        candidate_tensors.append(preprocess(img).unsqueeze(0))
    candidate_tensors = torch.cat(candidate_tensors, dim=0).to(device)
    # 编码文本
    text_tokens = clip.tokenize(query_text, truncate=True).to(device)

    with torch.no_grad():
        ref_feat = F.normalize(clip_model.encode_image(ref_img_tensor), dim=-1)
        text_feat = F.normalize(clip_model.encode_text(text_tokens), dim=-1)
        query_feat = generator(text_feat, ref_feat)   # [1, D]
        cand_feats = F.normalize(clip_model.encode_image(candidate_tensors), dim=-1)
        sim = (query_feat @ cand_feats.T).squeeze(0) / temperature
        top_indices = sim.argsort(descending=True)[:top_k]
    return [(candidate_image_paths[i], sim[i].item()) for i in top_indices]

# ==================== 加载提示库 ====================
def load_concept_extensions(json_path, npz_path):
    with open(json_path, 'r') as f:
        concepts = json.load(f)   # list of dict
    npz = np.load(npz_path)

    concept_names = []
    concept_name_embs = []
    concept_extend_embs = []   # list of list of tensors
    for item in concepts:
        name = item['name']
        name_key = item['name_emb_key']
        name_emb = torch.from_numpy(npz[name_key]).float()

        # 扩展描述的 embedding 列表
        extend_keys = item['extend_desc_emb_key']
        extend_embs = [torch.from_numpy(npz[key]).float() for key in extend_keys]

        concept_names.append(name)
        concept_name_embs.append(name_emb)
        concept_extend_embs.append(extend_embs)

    concept_name_embs = torch.stack(concept_name_embs, dim=0)  # [C, D]
    return concept_names, concept_name_embs, concept_extend_embs

# -------------------- 主函数 --------------------
def main():
    # 1. 构建训练三元组和验证数据
    # 获取所有车辆 ID
    with open(config.track_ann_file, 'r') as f:
        tracks = json.load(f)
    all_track_ids = list(tracks.keys())
    random.shuffle(all_track_ids)
    split_idx = int(len(all_track_ids) * 0.8)   # 80% 训练，20% 验证
    train_track_ids = set(all_track_ids[:split_idx])
    val_track_ids = set(all_track_ids[split_idx:])
    print(f"Total tracks: {len(all_track_ids)}, Train tracks: {len(train_track_ids)}, Val tracks: {len(val_track_ids)}")
    print("Building training triplets...")
    train_triplets = build_train_triplets(config.track_ann_file, config.image_root, allowed_track_ids=train_track_ids)
    print(f"Number of training triplets: {len(train_triplets)}")
    print("Building validation data...")
    candidate_images, val_queries = build_validation_data(config.track_ann_file, config.image_root, val_track_ids, 3)
    print(f"Validation candidates: {len(candidate_images)}, queries: {len(val_queries)}")
    # 2. 创建 Dataset 和 DataLoader
    train_dataset = TripletDataset(train_triplets, preprocess)
    # 注意：训练时使用普通的随机采样即可，对比学习会利用 batch 内的负样本
    train_sampler = TrackMutualSampler(train_triplets, config.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                          sampler=train_sampler, num_workers=config.num_workers,
                          pin_memory=True, drop_last=True)
    val_dataset = ValidationDataset(candidate_images, val_queries, preprocess,
                                    cache_path=os.path.join(config.save_dir, 'candidate_feats.pt'))
    # 验证时不使用 DataLoader 的 batch，因为需要逐一查询并检索整个候选集，我们直接在 evaluate 中遍历

    # 3. 模型和优化器
    concept_names, concept_name_embs, concept_extend_embs = load_concept_extensions(
        'concept_extend.json', 'concept_extend.embeddings.npz'
    )
    # 创建生成器
    # 直接利用语义相似度，基于二阶段的零样本检索，
    # generator = SemanticPromptGenerator(concept_name_embs, concept_extend_embs, temperature=0.07).to(device)
    # # 可选：将 temperature 设为可学习
    # generator.temperature.requires_grad = True
    # optimizer = torch.optim.Adam([generator.temperature], lr=0.01)  # 只训练温度
    # 结合扩展语义与可学习 MLP
    generator = PromptGenerator(concept_vectors, concept_extend_embs, clip_dim, num_concepts, config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)

    # 4. 训练循环
    best_map = 0.0
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_epoch(clip_model, generator, train_loader, optimizer, config.device, config.temperature)
        print(f"Train Loss: {train_loss:.4f}")

        # 每 5 个 epoch 验证一次
        if epoch % 5 == 0:
            recalls, mAP = evaluate_batched(clip_model, generator, val_dataset, config.device, config.temperature)
            print(f"Validation Results: R@1={recalls[1]:.2f}, R@5={recalls[5]:.2f}, R@10={recalls[10]:.2f}, MRR={mAP:.2f}")
            # mrr是什么指标，如果要计算map，该怎么修改
            if mAP > best_map:
                best_map = mAP
                torch.save(generator.state_dict(), os.path.join(config.save_dir, 'best_generator.pth'))
                print("Best model saved.")

    print("Training finished.")

if __name__ == '__main__':
    main()