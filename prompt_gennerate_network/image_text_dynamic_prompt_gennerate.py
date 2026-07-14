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
import pickle

# -------------------- 配置 --------------------
class Config:
    # 数据路径（请根据实际情况修改）
    data_root = '../../dataset/cityflow-nl/'            # 原始数据根目录（包含 train-tracks.json 等）
    image_root = '../../dataset/cityflow-nl/'     # 提取的图像根目录（包含 S01, S03 等）
    track_ann_file = os.path.join(data_root, 'train_tracks.json')   # 车辆轨迹标注
    prompt_library_root = '../prompt_library/result/'
    concept_extend_file = os.path.join(prompt_library_root, 'concept_extend_expand.json')
    concept_extend_embeddings = os.path.join(prompt_library_root, 'concept_extend_expand.embeddings.npz')
    # CLIP 模型
    clip_model_name = 'ViT-B/32'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 概念向量库
    concept_vector_path = './concept_vectors.npy'
    # 训练参数
    batch_size =64
    epochs = 50
    lr = 1e-5
    temperature = 0.07
    hidden_dim = 512
    num_workers = 4
    # 保存路径
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    # 验证时候选集大小（-1 表示使用全部候选）
    val_candidate_size = -1

config = Config()
torch.set_default_dtype(torch.float32)

# -------------------- 加载 CLIP 和概念向量 --------------------
print("Loading CLIP model...")
clip_model, preprocess = clip.load(config.clip_model_name, device=config.device)
for param in clip_model.parameters():
    param.requires_grad = False
clip_dim = clip_model.visual.output_dim

# print("Loading concept vectors...")
# concept_vectors = np.load(config.concept_vector_path)
# concept_vectors = torch.from_numpy(concept_vectors).float().to(config.device)
# concept_vectors = F.normalize(concept_vectors, dim=-1)
# num_concepts = concept_vectors.size(0)

def get_or_create_track_split(track_ann_file, split_file, train_ratio=0.8, seed=42):
    """
    如果 split_file 存在，则加载并返回 train_track_ids, val_track_ids；
    否则，从 track_ann_file 读取所有 track_id，按比例随机划分，保存到 split_file，再返回。
    """
    if os.path.exists(split_file):
        print(f"Loading track split from {split_file}")
        with open(split_file, 'rb') as f:
            train_track_ids, val_track_ids = pickle.load(f)
        return train_track_ids, val_track_ids
    else:
        print(f"Creating new track split, saving to {split_file}")
        with open(track_ann_file, 'r') as f:
            tracks = json.load(f)
        all_track_ids = list(tracks.keys())
        random.seed(seed)
        random.shuffle(all_track_ids)
        split_idx = int(len(all_track_ids) * train_ratio)
        train_track_ids = set(all_track_ids[:split_idx])
        val_track_ids = set(all_track_ids[split_idx:])
        # 保存为 set 列表 (pickle 支持 set)
        with open(split_file, 'wb') as f:
            pickle.dump((train_track_ids, val_track_ids), f)
        return train_track_ids, val_track_ids

def compute_prior_scores_single(query_feat, concept_embs):
    """
    计算查询特征与每个概念名称/描述的相似度
    query_feat: [B, D]
    concept_embs: [C, D]  单个向量 per concept
    returns: [B, C] 余弦相似度
    """
    # 确保都是 float32 并归一化
    query_feat = query_feat.float()
    concept_embs = concept_embs.float()
    # 直接点积（因为已经归一化）
    scores = query_feat @ concept_embs.T
    return scores

def compute_prior_scores(query_feat, concept_extend_embs):
    """
    计算查询特征与每个概念扩展描述的相似度（取最大值）
    返回 float32 类型的得分矩阵
    """
    B = query_feat.size(0)
    C = len(concept_extend_embs)
    device = query_feat.device
    # 强制使用 float32
    scores = torch.zeros(B, C, device=device, dtype=torch.float32)
    for c, exts in enumerate(concept_extend_embs):
        if len(exts) == 0:
            continue
        # 将扩展描述堆叠并转换为 float32
        ext_mat = torch.stack(exts, dim=0).to(device=device, dtype=torch.float32)
        # 确保 query_feat 也是 float32
        q = query_feat.float()  # 如果已经是 float32，此操作无影响
        sim = q @ ext_mat.T
        scores[:, c] = sim.max(dim=1)[0]
    return scores

# -------------------- PromptGenerator --------------------
class PromptGenerator(nn.Module):
    def __init__(self, concept_name_embs, concept_extend_embs, concept_extend_mean_embs, clip_dim, num_concepts, hidden_dim=256):
        super().__init__()
        self.gamma_list = []
        # # 确保概念名称向量为 float32 并注册为 buffer
        # self.register_buffer('concept_names', concept_name_embs.float())
        # 扩展描述向量列表（每个元素已是 float32）
        self.concept_name_embs = concept_name_embs

        # 扩展描述向量列表（每个元素已是 float32）
        self.concept_extend_embs = concept_extend_embs

        self.concept_extend_mean_embs = concept_extend_mean_embs

        # 可学习 MLP：输入是 先验得分 + 原始查询特征
        self.mlp = nn.Sequential(
            nn.Linear(num_concepts + clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.res_scale = nn.Parameter(torch.FloatTensor([0.1]))

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

    def forward(self, text_feat, img_feat=None):
        # 将输入的文本和图像特征统一转为 float32
        text_feat = text_feat.float()
        if img_feat is not None:
            img_feat = img_feat.float()

        # 1. 融合文本和图像得到查询特征
        if img_feat is not None:
            query_feat = F.normalize(text_feat + img_feat, dim=-1)
        else:
            query_feat = text_feat

        # 2. 计算先验得分（返回 float32）
        prior = compute_prior_scores(query_feat, self.concept_extend_embs)   # [B, C]

        # 3. 将先验得分与原始查询特征拼接，输入 MLP 得到调整后的 logits
        mlp_input = torch.cat([prior, query_feat], dim=-1)   # [B, C + D]
        logits = self.mlp(mlp_input)                         # [B, C]

        # 4. 应用温度和 softmax
        weights = F.softmax(logits / self.temperature, dim=-1)

        # 5. 加权组合概念名称向量生成提示
        dyn_prompt = weights @ self.concept_extend_mean_embs            # [B, D]
        dyn_prompt_norm = F.normalize(dyn_prompt, p=2, dim=-1)

        # 假设通过一个线性层学习门控信号
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
        # combined = F.normalize(combined, dim=-1)
        return combined, weights

    # 🛠️ 新增方法：清空上一轮的 gamma 记录
    def reset_gamma_tracking(self):
        self.gamma_list = []

    # 🛠️ 新增方法：计算整轮 Epoch 的 Gamma 平均值
    def get_average_gamma(self):
        if not self.gamma_list:
            return 0.0
        # 将所有 Batch 的 gamma 拼接并求全局平均值
        return torch.cat(self.gamma_list, dim=0).mean().item()

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
        # 扩大批次，每个track_取5个
        num_pairs_per_caption = 5
        for cap in captions:
            for _ in range(num_pairs_per_caption):
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

def build_validation_data(track_ann_file, image_root, val_track_ids, num_targets=2, sample_print=True, num_samples=5, cache_file=None):
    """
    划分验证集车辆，构建：
        candidate_images: 所有验证车辆的全部图像路径列表
        queries: 每个查询包含 ref_img_path, caption, target_img_path (在 candidate_images 中的索引)

    Args:
        cache_file: 缓存文件路径，如果提供且存在，则直接加载缓存；否则构建并保存
    """
    # 如果提供了缓存文件且存在，则直接加载
    if cache_file is not None and os.path.exists(cache_file):
        print(f"Loading cached validation data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        candidate_images = cached_data['candidate_images']
        queries = cached_data['queries']

        # 如果 sample_print 为 True，打印一些样本信息
        if sample_print and len(queries) > 0:
            print("\n=== Random samples from validation queries (loaded from cache) ===")
            sample_queries = random.sample(queries, min(num_samples, len(queries)))
            for i, q in enumerate(sample_queries):
                print(f"\nSample {i+1}:")
                print(f"  Track ID: {q['track_id']}")
                print(f"  Caption: {q['caption']}")
                print(f"  Reference image: {q['ref_img']}")
                print(f"  Target image(s):")
                for idx in q['target_idxs']:
                    print(f"    - {candidate_images[idx]}")
                print("  ---")

        return candidate_images, queries

    # 否则，正常构建
    print("Building validation data...")
    with open(track_ann_file, 'r') as f:
        tracks = json.load(f)

    seen_images = set()
    candidate_images = []
    candidate_track_ids = []
    img_to_idx = {}

    for tid in val_track_ids:
        if tid not in tracks:
            continue
        frames = tracks[tid]['frames']
        for frame in frames:
            candidate_track_ids.append(tid)
            img_path = os.path.join(image_root, frame.lstrip('./'))
            if img_path not in seen_images:
                seen_images.add(img_path)
                candidate_images.append(img_path)
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
            ref_idx = img_to_idx.get(ref_full) # 获取参考图在全库中的索引
            target_idxs = []
            for target_img in target_frames:
                target_full = os.path.join(image_root, target_img.lstrip('./'))
                if target_full in img_to_idx:
                    target_idxs.append(img_to_idx[target_full])
            if not target_idxs:
                continue
            queries.append({
                'ref_img': ref_full,
                'ref_idx': ref_idx,       # 新增：把参考图的索引也存进 query
                'caption': cap,
                'target_idxs': target_idxs,   # 存储多个索引
                'track_id': tid
            })

    # 保存缓存
    if cache_file is not None:
        print(f"Saving validation data cache to {cache_file}")
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'candidate_images': candidate_images,
                'queries': queries
            }, f)

    return candidate_images, queries

# -------------------- 数据集类 --------------------
class TripletDataset(Dataset):
    def __init__(self, triplets, preprocess, track_to_int=None):
        self.triplets = triplets
        self.preprocess = preprocess
        self.track_to_int = track_to_int

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        ref_img = Image.open(item['ref_img']).convert('RGB')
        target_img = Image.open(item['target_img']).convert('RGB')
        ref_img = self.preprocess(ref_img)
        target_img = self.preprocess(target_img)
        text = clip.tokenize(item['caption']).squeeze(0)
        track_id = item['track_id']
        if self.track_to_int is not None:
            track_id = self.track_to_int[track_id]   # 转为整数
        return ref_img, target_img, text, track_id   # 增加 track_id

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
            if self.candidate_feats.shape[0] != len(self.candidate_images):
                print(f"⚠️ 警告: 检测到缓存特征数 ({self.candidate_feats.shape[0]}) 与当前候选图片数 ({len(self.candidate_images)}) 不匹配！")
                print(f"这极有可能是由于数据集划分改变或使用了旧的 Baseline 缓存导致的。")
                print(f"系统将无视并自动删除旧缓存，重新提取特征以确保实验正确性...")
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
        text = clip.tokenize(query['caption']).squeeze(0)
        return ref_img, text, query['target_idxs'], query['track_id']

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
        track_ids = self.track_ids.copy()
        if self.shuffle:
            random.shuffle(track_ids)

        batch = []
        for tid in track_ids:
            idx = random.choice(self.track_to_indices[tid])
            batch.append(idx)

            # 满一个 batch 就输出
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # 【修复】最后剩下的样本也输出（测试模式关键）
        if len(batch) > 0:
            yield batch

    def __len__(self):
        # 【修复】向上取整，绝对不会返回 0
        return (len(self.track_ids) + self.batch_size - 1) // self.batch_size

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


# -------------------- 训练函数 --------------------
def train_epoch(clip_model, generator, dataloader, optimizer, device, temperature):
    clip_model.eval()
    generator.train()
    total_loss = 0
    num_batches = 0
    for ref_imgs, target_imgs, texts, track_ids in tqdm(dataloader, desc='Training'):
        ref_imgs = ref_imgs.to(device)
        target_imgs = target_imgs.to(device)
        texts = texts.to(device)
        batch_size = ref_imgs.size(0)

        with torch.no_grad():
            # 编码并归一化，然后转换为 float32
            ref_feat = clip_model.encode_image(ref_imgs)
            ref_feat = F.normalize(ref_feat, dim=-1).float()
            target_feat = clip_model.encode_image(target_imgs)
            target_feat = F.normalize(target_feat, dim=-1).float()
            text_feat = clip_model.encode_text(texts)
            text_feat = F.normalize(text_feat, dim=-1).float()

        # 生成查询特征（生成器内部已做类型转换，但再次确保）
        query_feat, weights = generator(text_feat, ref_feat)
        query_feat = query_feat.float()

        # 计算相似度矩阵
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
    ndcg_sum = {5: 0.0, 10: 0.0}   # 新增 NDCG@5, NDCG@10

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
            texts.append(clip.tokenize(q['caption']).squeeze(0))
            target_idxss.append(q['target_idxs'])   # 存储列表

        ref_imgs = torch.stack(ref_imgs).to(device)
        texts = torch.stack(texts).to(device)

        ref_feat = F.normalize(clip_model.encode_image(ref_imgs), dim=-1)
        text_feat = F.normalize(clip_model.encode_text(texts), dim=-1)
        query_feat, weights = generator(text_feat, ref_feat)   # [B, D]

        # 在 evaluate_batched 函数内部，获取候选特征后添加：
        candidate_feats = candidate_feats.float()
        # 在得到 query_feat 后也添加：
        query_feat = query_feat.float()
        sim = query_feat @ candidate_feats.T / temperature   # [B, C]

        for i in range(len(batch_queries)):
            # sim_i = sim[i]
            sim_i = sim[i].clone() # 克隆一行相似度，避免原地修改干扰并行计算
            # 【核心改动】：获取当前 Query 的参考图索引，强制将其相似度降到极低
            ref_idx = batch_queries[i].get('ref_idx')
            if ref_idx is not None:
                sim_i[ref_idx] = -1e9

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
    # 平均指标
    for k in k_list:
        recalls[k] = recalls[k] / num_queries * 100
    mAP = ap_sum / num_queries * 100
    return recalls, mAP, ndcg_sum

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
    text_tokens = clip.tokenize(query_text).to(device)

    with torch.no_grad():
        ref_feat = F.normalize(clip_model.encode_image(ref_img_tensor), dim=-1)
        text_feat = F.normalize(clip_model.encode_text(text_tokens), dim=-1)
        query_feat, weights = generator(text_feat, ref_feat)   # [1, D]
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
    concept_desc_embs = []
    concept_extend_embs = []   # list of list of tensors
    concept_extend_mean_embs = []   # list of list of tensors
    for item in concepts:
        name = item['name']
        name_key = item['name_emb_key']
        name_emb = torch.from_numpy(npz[name_key]).float()
        # 修复：去除多余的维度，确保 shape 为 (D,)
        if name_emb.dim() == 2:
            if name_emb.size(0) == 1:
                name_emb = name_emb.squeeze(0)
            elif name_emb.size(1) == 1:
                name_emb = name_emb.squeeze(1)
        # 如果依然是二维但 size(0) 和 size(1) 都不为 1，则报错或取均值等
        if name_emb.dim() != 1:
            raise ValueError(f"Unexpected shape for {name_key}: {name_emb.shape}")

        desc_key = item['desc_emb_key']
        desc_emb = torch.from_numpy(npz[desc_key]).float()
        # 修复：去除多余的维度，确保 shape 为 (D,)
        if desc_emb.dim() == 2:
            if desc_emb.size(0) == 1:
                desc_emb = desc_emb.squeeze(0)
            elif desc_emb.size(1) == 1:
                desc_emb = desc_emb.squeeze(1)
        # 如果依然是二维但 size(0) 和 size(1) 都不为 1，则报错或取均值等
        if desc_emb.dim() != 1:
            raise ValueError(f"Unexpected shape for {desc_key}: {desc_emb.shape}")


        extend_keys = item['extend_desc_emb_key']
        extend_embs = []
        for key in extend_keys:
            emb = torch.from_numpy(npz[key]).float()
            # 同样处理扩展描述的维度
            if emb.dim() == 2:
                if emb.size(0) == 1:
                    emb = emb.squeeze(0)
                elif emb.size(1) == 1:
                    emb = emb.squeeze(1)
            extend_embs.append(emb)

        extend_mean_keys = item['desc_mean_emb_key']
        extend_mean_embs = torch.from_numpy(npz[extend_mean_keys]).float()
        # 修复：去除多余的维度，确保 shape 为 (D,)
        if extend_mean_embs.dim() == 2:
            if extend_mean_embs.size(0) == 1:
                extend_mean_embs = extend_mean_embs.squeeze(0)
            elif extend_mean_embs.size(1) == 1:
                extend_mean_embs = extend_mean_embs.squeeze(1)
        # 如果依然是二维但 size(0) 和 size(1) 都不为 1，则报错或取均值等
        if extend_mean_embs.dim() != 1:
            raise ValueError(f"Unexpected shape for {desc_key}: {extend_mean_embs.shape}")

        concept_names.append(name)
        concept_name_embs.append(name_emb)
        concept_desc_embs.append(desc_emb)
        concept_extend_embs.append(extend_embs)
        concept_extend_mean_embs.append(extend_mean_embs)

    concept_name_embs = torch.stack(concept_name_embs, dim=0).to(config.device)  # [C, D]
    concept_desc_embs = torch.stack(concept_desc_embs, dim=0).to(config.device)  # [C, D]
    concept_extend_mean_embs = torch.stack(concept_extend_mean_embs, dim=0).to(config.device)  # [C, D]
    return concept_names, concept_name_embs, concept_extend_embs, concept_desc_embs, concept_extend_mean_embs

def visualize_concept_weights(generator, is_generator, val_dataset, concept_names, device, num_samples=10, top_k=10):
    """
    随机抽取 num_samples 个验证集查询，打印每个查询的 caption、ref_img 路径，
    以及模型预测的权重最高的 top_k 个概念。
    """
    generator.eval()
    queries = val_dataset.queries
    indices = random.sample(range(len(queries)), min(num_samples, len(queries)))

    for idx in indices:
        q = queries[idx]
        caption = q['caption']
        ref_img_path = q['ref_img']

        # 加载参考图像并预处理
        ref_img = Image.open(ref_img_path).convert('RGB')
        ref_tensor = preprocess(ref_img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(caption).to(device)

        with torch.no_grad():
            ref_feat = F.normalize(clip_model.encode_image(ref_tensor), dim=-1).float()
            text_feat = F.normalize(clip_model.encode_text(text_tokens), dim=-1).float()
            if is_generator:
                _, weights = generator(text_feat, ref_feat)   # weights shape: [1, C]
            else:
                _ = generator(text_feat, ref_feat)
        # 获取权重最高的 top_k 个概念的索引

        print("\n" + "="*60)
        print(f"Query: {caption}")
        print(f"Reference Image: {ref_img_path}")
        if is_generator:
            weights_np = weights.squeeze(0).cpu().numpy()
            top_indices = weights_np.argsort()[-top_k:][::-1]
            print(f"Top-{top_k} activated concepts:")
            for i, idx_c in enumerate(top_indices):
                concept_name = concept_names[idx_c]
                weight_val = weights_np[idx_c]
                print(f"  {i+1}. {concept_name}: {weight_val:.4f}")

def validate_with_model(model_path, config, device, val_track_ids=None, val_querys_path=None, batch_size=64, k_list=[1,5,10]):
    """
    加载已训练好的 PromptGenerator 模型，并在验证集上进行检索评估。

    Args:
        model_path (str): 保存的模型权重路径（.pth 文件）
        config (Config): 配置对象，需包含以下属性：
            - track_ann_file: 轨迹标注文件路径
            - image_root: 图像根目录
            - concept_extend_file: 概念扩展 JSON 文件
            - concept_extend_embeddings: 概念扩展 embedding npz 文件
            - clip_model_name: CLIP 模型名称
            - device: 设备（'cuda' 或 'cpu'）
            - temperature: 损失中使用的温度（用于评估）
            - save_dir: 保存缓存文件的目录
        device (torch.device): 推理设备
        val_track_ids (set or list, optional): 验证集车辆 ID 列表。若不提供，则尝试从缓存划分文件加载
        batch_size (int): 评估时的批处理大小
        k_list (list): 要计算的 Recall@K 列表

    Returns:
        dict: 包含各 Recall 值和 mAP 的字典
    """
    # 1. 加载 CLIP 模型（与训练时相同）
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(config.clip_model_name, device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()

    # 2. 加载概念扩展库
    print("Loading concept extensions...")
    concept_names, concept_name_embs, concept_extend_embs, concept_desc_embs, concept_extend_mean_embs = load_concept_extensions(
        config.concept_extend_file, config.concept_extend_embeddings
    )
    # 创建生成器（与训练时结构一致）
    generator = PromptGenerator(
        concept_name_embs, concept_extend_embs, concept_extend_mean_embs,
        clip_model.visual.output_dim, len(concept_names), config.hidden_dim
    ).to(device)
    # 加载权重
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    # 3. 构建验证数据集
    # 若未提供 val_track_ids，尝试从之前保存的划分文件中加载
    if val_track_ids is None:
        split_file = os.path.join(config.save_dir, 'track_split.pkl')
        if os.path.exists(split_file):
            with open(split_file, 'rb') as f:
                _, val_track_ids = pickle.load(f)
            print(f"Loaded val track IDs from {split_file}: {len(val_track_ids)} tracks")
        else:
            raise FileNotFoundError(f"Split file {split_file} not found. Please provide val_track_ids manually.")

    # 构建验证查询和候选图像列表
    print("Building validation data...")
    candidate_images, val_queries = build_validation_data(
        config.track_ann_file, config.image_root, val_track_ids,
        num_targets=3, sample_print=True, cache_file=os.path.join(config.save_dir, 'clip4cir_validation_cache.pkl')
    )

    if val_querys_path is not None:
        if os.path.exists(val_querys_path):
            with open(val_querys_path, 'rb') as f:
                val_queries = json.load(f)
    print(f"Validation candidates: {len(candidate_images)}, queries: {len(val_queries)}")

    # 创建 ValidationDataset（会自动缓存候选特征，避免重复提取）
    val_dataset = ValidationDataset(
        candidate_images, val_queries, preprocess,
        cache_path=os.path.join(config.save_dir, 'candidate_feats_clip4cir.pt')
    )

    # 随机从验证集抽取图像查看解释情况 验证模型时注释，查看解释性分类打开
    # visualize_concept_weights(generator, true, val_dataset, concept_names, device, num_samples=3, top_k=10)

    # 4. 评估
    print("Starting evaluation...")
    recalls, mAP, ndcg_sum = evaluate_batched(
        clip_model, generator, val_dataset, device,
        temperature=config.temperature, batch_size=batch_size, k_list=k_list
    )

    # 打印结果
    print("\n========== Validation Results ==========")
    for k in k_list:
        print(f"Recall@{k}: {recalls[k]:.2f}%")
    print(f"mAP: {mAP:.2f}%")
    print(f"NDCG@5: {ndcg_sum[5]:.2f}%, NDCG@10: {ndcg_sum[10]:.2f}%")
    print("========================================")

    # 返回结果字典
    results = {f"R@{k}": recalls[k] for k in k_list}
    results["mAP"] = mAP
    return results

# -------------------- 主函数 --------------------
def main():
    # 1. 构建训练三元组和验证数据
    # -------------------- 测试模式开关 --------------------
    test_mode = False   # 改为 False 关闭测试模式
    if test_mode:
        print("=== TEST MODE ENABLED: using small subset of data ===")
        test_track_limit = 20       # 只取前20个车辆
        config.epochs = 1           # 只训练1个epoch
        config.batch_size = 8       # 减小batch size
        config.num_workers = 0      # 避免多进程问题
    # 获取所有车辆 ID
    split_file = os.path.join(Config.save_dir, 'track_split.pkl')   # 保存到 checkpoints 目录下
    train_track_ids, val_track_ids = get_or_create_track_split(
        Config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )
    print(f"Train tracks: {len(train_track_ids)}, Val tracks: {len(val_track_ids)}")
    print("Building training triplets...")
    train_triplets = build_train_triplets(config.track_ann_file, config.image_root, allowed_track_ids=train_track_ids)
    if test_mode:
        # 限制训练三元组数量，加快测试
        train_triplets = train_triplets[:200]
    if test_mode:
        all_track_ids = train_track_ids[:test_track_limit]
    else:
        all_track_ids = sorted(set([t['track_id'] for t in train_triplets]))  # 排序保证确定性
    track_to_int = {tid: idx for idx, tid in enumerate(all_track_ids)}
    print(f"Number of training triplets: {len(train_triplets)}")
    print("Building validation data...")
    candidate_images, val_queries = build_validation_data(
        config.track_ann_file, config.image_root, val_track_ids,
        num_targets=3, sample_print=True, cache_file=os.path.join(config.save_dir, 'clip4cir_validation_cache.pkl')
    )
    if test_mode:
        # 限制验证集大小
        candidate_images = candidate_images[:100]
        val_queries = val_queries[:10]
    print(f"Validation candidates: {len(candidate_images)}, queries: {len(val_queries)}")
    # 2. 创建 Dataset 和 DataLoader
    # train_dataset = TripletDataset(train_triplets, preprocess)
    # # 注意：训练时使用普通的随机采样即可，对比学习会利用 batch 内的负样本
    # sampler = TrackMutualSampler(train_triplets, batch_size=config.batch_size, shuffle=True)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=sampler,  # 必须是 batch_sampler，不是 sampler！
    #     num_workers=config.num_workers,
    #     pin_memory=True
    # )
    # 放弃批内互斥采样，使用普通随机采样
    train_dataset = TripletDataset(train_triplets, preprocess, track_to_int)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataset = ValidationDataset(candidate_images, val_queries, preprocess,
                                    cache_path=os.path.join(config.save_dir, 'candidate_feats_clip4cir.pt'))
    # 验证时不使用 DataLoader 的 batch，因为需要逐一查询并检索整个候选集，我们直接在 evaluate 中遍历

    # 3. 模型和优化器
    concept_names, concept_name_embs, concept_extend_embs, concept_desc_embs, concept_extend_mean_embs = load_concept_extensions(
        config.concept_extend_file, config.concept_extend_embeddings
    )
    # 创建生成器
    # 直接利用语义相似度，基于二阶段的零样本检索，
    # generator = SemanticPromptGenerator(concept_name_embs, concept_extend_embs, temperature=0.07).to(device)
    # # 可选：将 temperature 设为可学习
    # generator.temperature.requires_grad = True
    # optimizer = torch.optim.Adam([generator.temperature], lr=0.01)  # 只训练温度
    # 结合扩展语义与可学习 MLP
    generator = PromptGenerator(concept_name_embs, concept_extend_embs, concept_extend_mean_embs, clip_dim,  len(concept_names), config.hidden_dim).to(config.device)
    # optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)
    skip_ids = {id(p) for p in list(generator.feature_combiner.parameters()) + list(generator.gate_layer.parameters())}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in generator.parameters() if id(p) not in skip_ids], 'lr': config.lr},
        {'params': generator.feature_combiner.parameters(), 'lr': 1e-4},
        {'params': generator.gate_layer.parameters(), 'lr': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 50个epoch

    # 4. 训练循环
    best_map = 0.0
    patience = 5
    early_stop_count = 0
    model_name= "generator_gates_enhance_image_expand_checkpoint.pth"
    # 断点文件路径
    ckpt_path = os.path.join(Config.save_dir, f'resume_{model_name}')
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
        train_loss = train_epoch(clip_model, generator, train_loader, optimizer, config.device, config.temperature)
        avg_gamma = generator.get_average_gamma()
        if avg_gamma > 0.0: # 规避消融了知识库导致没算gamma的情况
            print(f"📊 [Gamma Monitor] Image Weight (Gamma): {avg_gamma:.4f}")
        print(f"Train Loss: {train_loss:.4f}")
        torch.save(generator.state_dict(), os.path.join(config.save_dir, f'train_{model_name}'))

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
        if not test_mode and epoch % 2 == 0:
            recalls, mAP, ndcg_sum = evaluate_batched(clip_model, generator, val_dataset, config.device, config.temperature)
            print(f"Validation Results: R@1={recalls[1]:.2f}, R@5={recalls[5]:.2f}, R@10={recalls[10]:.2f}, MRR={mAP:.2f}")
            print(f"NDCG@5: {ndcg_sum[5]:.2f}%, NDCG@10: {ndcg_sum[10]:.2f}%")
            # mrr是什么指标，如果要计算map，该怎么修改
            if mAP > best_map:
                best_map = mAP
                early_stop_count = 0
                torch.save(generator.state_dict(), os.path.join(Config.save_dir, f'best_{model_name}'))
                print("New Best model saved!")
            else:
                early_stop_count += 1
                print(f"No improve, early_stop count: {early_stop_count}/{patience}")

        # 早停判断
        if early_stop_count >= patience:
            print(f"Early Stop Trigger! {patience} epochs no mAP improve, exit training.")
            break

        scheduler.step()
        print("Training finished.")

if __name__ == '__main__':
    # main()
    # 假设 config 已经按原代码配置好
    config = Config()
    device = torch.device(config.device)
    model_path = "./checkpoints/best_generator_gates_enhance_image_expand_checkpoint.pth"
    val_querys_path = "./class-evaluate/special_weather.json"
    results = validate_with_model(model_path, config, device, val_querys_path=val_querys_path)
