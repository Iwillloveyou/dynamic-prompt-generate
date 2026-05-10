import json
import random
from typing import List, Dict
import os

#将cityflow-nl.json的图文对按照训练和测试阶段分别构建正负样本对

def load_annotation(json_path: str, test_mode: bool = False) -> Dict:
    """加载JSON标注文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # test模式：只保留前10条样本，快速验证
    if test_mode:
        data = dict(list(data.items())[:15])
        print("🔍 TEST模式：仅使用前10条样本进行验证")
    return data

def split_into_batches(sample_keys: List[str], batch_size: int) -> List[List[str]]:
    """将所有样本ID按批次大小分组"""
    batches = []
    for i in range(0, len(sample_keys), batch_size):
        batches.append(sample_keys[i:i+batch_size])
    return batches

def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def construct_samples(anno_data: Dict, batch_size: int = 3, val_ratio: float = 0.2) -> List[Dict]:
    """构造 frame-nl-frame 正负样本（每条样本仅含 1 个 frame 字段）"""
    all_sample_ids = list(anno_data.keys())
    if len(all_sample_ids) < batch_size:
        raise ValueError(f"样本总数({len(all_sample_ids)})小于批次大小({batch_size})")

    random.shuffle(all_sample_ids)
    val_num = max(1, int(len(all_sample_ids) * val_ratio))
    val_sample_ids = set(all_sample_ids[:val_num])
    train_sample_ids = set(all_sample_ids[val_num:])

    batches = split_into_batches(all_sample_ids, batch_size)
    train_samples = []
    val_samples = []

    for batch_idx, batch_sample_ids in enumerate(batches):
        if len(batch_sample_ids) < batch_size:
            continue


        for current_id in batch_sample_ids:
            current_data = anno_data[current_id]
            current_frames = current_data["frames"]
            current_nls = current_data["nl"]

            if len(current_frames) < 3 or len(current_nls) < 1:
                continue

            pos_frames = random.sample(current_frames, 3)  # 选3帧
            pos_nl = random.sample(current_nls, 3)

            # 验证集
            if current_id in val_sample_ids:
                for frame in pos_frames:
                    # 验证集target_frames：查询帧排第一，其余按原顺序
                    target_frames = [frame] + [f for f in current_frames if f != frame]
                    for nl in pos_nl:
                        val_samples.append({
                            "batch_id": batch_idx,
                            "sample_id": current_id,
                            "split": "validation",
                            "type": "positive",
                            "frames": frame,  # 单个字符串，不是数组
                            "nl": nl,
                            "target_frames": target_frames
                        })
                continue

            # 训练集
            if current_id not in train_sample_ids:
                continue
            # 每条单独生成一条样本
            for frame in pos_frames:
                for nl in pos_nl:
                    # ===================== 正样本：拆成 3 条独立样本 =====================
                    train_samples.append({
                        "batch_id": batch_idx,
                        "sample_id": current_id,
                        "split": "train",
                        "type": "positive",
                        "frames": frame,  # 单个字符串，不是数组
                        "nl": nl,
                        "target_frame": random.choice(current_frames)
                    })

                    other_ids = [sid for sid in batch_sample_ids if sid != current_id]
                    if not other_ids:
                        continue
                    # ===================== 负样本：与批次内其它样本的图片是不相符的 =====================
                    for neg_other_id in other_ids:
                        neg_other_frames = anno_data[neg_other_id]["frames"]
                        neg_target = random.choice(neg_other_frames)
                        train_samples.append({
                            "batch_id": batch_idx,
                            "sample_id": current_id,
                            "split": "train",
                            "type": "negative",
                            "frames": frame,  # 单个字符串，不是数组
                            "nl": nl,
                            "target_frame": neg_target
                        })
    return train_samples, val_samples

def save_samples(samples: List[Dict], save_path: str):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

# ===================== 主运行 =====================
if __name__ == "__main__":
    ANNOTATION_PATH = "train-tracks.json"  # 你的标注文件
    BATCH_SIZE = 5
    VAL_RATIO = 0.2

    output_dir = "./annotations/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    anno = load_annotation(ANNOTATION_PATH, test_mode=True)
    train_data, val_data = construct_samples(anno, BATCH_SIZE, VAL_RATIO)

    # 分别保存到两个文件
    save_to_json(train_data, output_dir + "train.json")
    save_to_json(val_data, output_dir + "val.json")

    # 输出统计
    print(f"✅ 保存完成！")
    print(f"训练集 train.json：{len(train_data)} 条")
    print(f"验证集 val.json：{len(val_data)} 条")

    # 展示示例
    if train_data:
        print("\n==== 训练集示例 ====")
        print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
    if val_data:
        print("\n==== 验证集示例 ====")
        print(json.dumps(val_data[0], indent=2, ensure_ascii=False))