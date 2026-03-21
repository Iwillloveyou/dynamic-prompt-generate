# split_dataset.py
import json
import random
from sklearn.model_selection import train_test_split

def split_dataset(ann_file, output_dir, val_ratio=0.2, seed=42):
    with open(ann_file, 'r') as f:
        samples = json.load(f)

    # 按图像路径去重，确保同一图像的不同描述在同一个split
    image_to_samples = {}
    for s in samples:
        img_path = s['image']
        if img_path not in image_to_samples:
            image_to_samples[img_path] = []
        image_to_samples[img_path].append(s)

    images = list(image_to_samples.keys())

    # 划分训练集和验证集
    train_images, val_images = train_test_split(
        images, test_size=val_ratio, random_state=seed
    )

    # 构建新的标注
    train_samples = []
    val_samples = []

    for img in train_images:
        train_samples.extend(image_to_samples[img])
    for img in val_images:
        # 将一部分训练数据转为验证集
        for s in image_to_samples[img]:
            s['split'] = 'val'
            val_samples.append(s)

    # 保存
    with open(f'{output_dir}/train.json', 'w') as f:
        json.dump(train_samples, f, indent=2)

    with open(f'{output_dir}/val.json', 'w') as f:
        json.dump(val_samples, f, indent=2)

    # 保留测试集不变
    test_samples = [s for s in samples if s['split'] == 'test']
    with open(f'{output_dir}/test.json', 'w') as f:
        json.dump(test_samples, f, indent=2)

    print(f"训练集: {len(train_samples)} 样本, {len(train_images)} 图像")
    print(f"验证集: {len(val_samples)} 样本, {len(val_images)} 图像")
    print(f"测试集: {len(test_samples)} 样本")

if __name__ == '__main__':
    split_dataset(
        ann_file='./annotations/cityflow-nl.json',
        output_dir='./annotations',
        val_ratio=0.15
    )