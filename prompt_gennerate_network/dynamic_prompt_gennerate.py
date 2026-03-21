import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

# -------------------- 配置参数 --------------------
class Config:
    # 数据集路径（请根据实际情况修改）
    data_root = './cityflow-nl'           # 数据集根目录，包含 images 和 annotations.json
    image_dir = os.path.join(data_root, 'images')
    ann_file = os.path.join(data_root, 'annotations.json')

    # CLIP模型
    clip_model_name = 'ViT-B/32'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 概念向量库（.npy文件，形状 [num_concepts, clip_dim]）
    concept_vector_path = './concept_vectors.npy'

    # 训练参数
    batch_size = 32
    epochs = 50
    lr = 1e-4
    temperature = 0.07                # 对比损失的温度系数
    hidden_dim = 512                   # 生成网络隐藏层维度
    num_workers = 4

    # 保存路径
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

config = Config()

# -------------------- 加载CLIP模型 --------------------
print("Loading CLIP model...")
clip_model, preprocess = clip.load(config.clip_model_name, device=config.device)
# 冻结CLIP参数
for param in clip_model.parameters():
    param.requires_grad = False
clip_dim = clip_model.visual.output_dim   # 通常是512或768

# -------------------- 加载概念向量库 --------------------
print("Loading concept vectors...")
concept_vectors = np.load(config.concept_vector_path)   # shape [C, clip_dim]
concept_vectors = torch.from_numpy(concept_vectors).float().to(config.device)
# L2归一化，保证与CLIP特征空间一致
concept_vectors = F.normalize(concept_vectors, dim=-1)
num_concepts = concept_vectors.size(0)

# -------------------- 定义提示生成网络 --------------------
class PromptGenerator(nn.Module):
    def __init__(self, concept_vectors, clip_dim, num_concepts, hidden_dim=512):
        super().__init__()
        # 固定概念向量库
        self.register_buffer('concept_vectors', concept_vectors)   # [C, D]

        # 生成权重分布的多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )

        # 可学习的融合系数（标量）
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_feat):
        """
        text_feat: [batch, clip_dim]  原始CLIP文本特征（已归一化）
        返回: [batch, clip_dim]  融合后的文本特征（已归一化）
        """
        # 计算概念权重分布
        logits = self.mlp(text_feat)                # [batch, C]
        weights = F.softmax(logits, dim=-1)         # [batch, C]

        # 加权组合概念向量得到动态提示
        dyn_prompt = torch.einsum('bc,cd->bd', weights, self.concept_vectors)  # [batch, D]

        # 融合：原始特征 + 动态提示 * 可学习系数
        combined = text_feat + self.alpha * dyn_prompt

        # 再次归一化
        combined = F.normalize(combined, dim=-1)
        return combined

# -------------------- 数据集定义 --------------------
class CityFlowNLDataset(Dataset):
    """CityFlow-NL 数据集，每张图像可能对应多个描述，每个描述作为一个独立样本"""
    def __init__(self, ann_file, image_dir, clip_preprocess, split='train'):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        self.image_dir = image_dir
        self.preprocess = clip_preprocess
        self.samples = []

        # 假设标注格式为: [{"image": "image.jpg", "caption": "a car ...", "split": "train"}, ...]
        for item in data:
            if item['split'] == split:
                self.samples.append({
                    'image_path': os.path.join(image_dir, item['image']),
                    'caption': item['caption']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.preprocess(image)
        # 对文本进行tokenize
        text = clip.tokenize(sample['caption'], truncate=True).squeeze(0)   # [77]
        return image, text

# -------------------- 训练函数 --------------------
def train_epoch(model, generator, dataloader, optimizer, device, temperature):
    model.eval()        # CLIP冻结
    generator.train()

    total_loss = 0
    num_batches = 0

    for images, texts in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        texts = texts.to(device)
        batch_size = images.size(0)

        # 提取图像特征（冻结）
        with torch.no_grad():
            image_feat = model.encode_image(images)          # [batch, clip_dim]
            image_feat = F.normalize(image_feat, dim=-1)

        # 提取原始文本特征（冻结）
        with torch.no_grad():
            text_feat_raw = model.encode_text(texts)         # [batch, clip_dim]
            text_feat_raw = F.normalize(text_feat_raw, dim=-1)

        # 通过生成网络得到增强的文本特征
        text_feat_aug = generator(text_feat_raw)             # [batch, clip_dim]

        # 计算相似度矩阵
        logits = text_feat_aug @ image_feat.T / temperature  # [batch, batch]

        # 构造标签（对角线为正样本）
        labels = torch.arange(batch_size, device=device)

        # 双向对比损失
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        loss = (loss_t2i + loss_i2t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

# -------------------- 验证/测试函数 --------------------
@torch.no_grad()
def evaluate(model, generator, dataloader, device, temperature, k_list=[1,5,10]):
    model.eval()
    generator.eval()

    all_image_feat = []
    all_text_feat_aug = []

    for images, texts in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        texts = texts.to(device)

        # 图像特征
        image_feat = model.encode_image(images)
        image_feat = F.normalize(image_feat, dim=-1)
        all_image_feat.append(image_feat)

        # 增强文本特征
        text_feat_raw = model.encode_text(texts)
        text_feat_raw = F.normalize(text_feat_raw, dim=-1)
        text_feat_aug = generator(text_feat_raw)
        all_text_feat_aug.append(text_feat_aug)

    image_feat_all = torch.cat(all_image_feat, dim=0)   # [N_img, D]
    text_feat_all = torch.cat(all_text_feat_aug, dim=0) # [N_txt, D] 注意 N_txt = N_img (一对一)

    # 计算相似度矩阵
    logits = text_feat_all @ image_feat_all.T / temperature   # [N, N]

    # 计算召回率（文本检索图像）
    n = logits.size(0)
    labels = torch.arange(n, device=device)

    # 按行排序（文本到图像）
    idx = logits.argsort(dim=-1, descending=True)   # [N, N]
    ranks = torch.where(idx == labels.view(-1,1))[1]  # 每个文本的正样本所在列索引
    recall_t2i = {}
    for k in k_list:
        recall_t2i[f'R@{k}'] = (ranks < k).float().mean().item() * 100

    # 按列排序（图像到文本）
    idx_img = logits.T.argsort(dim=-1, descending=True)
    ranks_img = torch.where(idx_img == labels.view(-1,1))[1]
    recall_i2t = {}
    for k in k_list:
        recall_i2t[f'R@{k}'] = (ranks_img < k).float().mean().item() * 100

    # 平均召回
    recall_avg = {}
    for k in k_list:
        recall_avg[f'R@{k}'] = (recall_t2i[f'R@{k}'] + recall_i2t[f'R@{k}']) / 2

    return recall_t2i, recall_i2t, recall_avg

# -------------------- 主训练流程 --------------------
def main():
    # 数据集
    train_dataset = CityFlowNLDataset(config.ann_file, config.image_dir, preprocess, split='train')
    val_dataset = CityFlowNLDataset(config.ann_file, config.image_dir, preprocess, split='val')   # 假设有val
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 初始化提示生成网络
    generator = PromptGenerator(concept_vectors, clip_dim, num_concepts, config.hidden_dim).to(config.device)

    # 优化器
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)

    # 训练循环
    best_recall = 0.0
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_epoch(clip_model, generator, train_loader, optimizer, config.device, config.temperature)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        recall_t2i, recall_i2t, recall_avg = evaluate(clip_model, generator, val_loader, config.device, config.temperature)
        print(f"Val Text->Image: R@1={recall_t2i['R@1']:.2f}, R@5={recall_t2i['R@5']:.2f}, R@10={recall_t2i['R@10']:.2f}")
        print(f"Val Image->Text: R@1={recall_i2t['R@1']:.2f}, R@5={recall_i2t['R@5']:.2f}, R@10={recall_i2t['R@10']:.2f}")
        print(f"Val Avg: R@1={recall_avg['R@1']:.2f}, R@5={recall_avg['R@5']:.2f}, R@10={recall_avg['R@10']:.2f}")

        # 保存最佳模型
        if recall_avg['R@1'] > best_recall:
            best_recall = recall_avg['R@1']
            torch.save(generator.state_dict(), os.path.join(config.save_dir, 'best_generator.pth'))
            print("Best model saved.")

    print("Training finished.")

if __name__ == '__main__':
    main()