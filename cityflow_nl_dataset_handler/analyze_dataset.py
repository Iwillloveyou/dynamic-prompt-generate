# analyze_dataset.py
import json
from collections import Counter

def analyze_dataset(ann_file):
    with open(ann_file, 'r') as f:
        samples = json.load(f)

    # 基本统计
    print(f"总样本数: {len(samples)}")

    # 按split统计
    split_counts = Counter([s['split'] for s in samples])
    print(f"按split分布: {dict(split_counts)}")

    # 描述长度统计
    desc_lengths = [len(s['caption'].split()) for s in samples]
    print(f"描述长度 - 平均: {sum(desc_lengths)/len(desc_lengths):.1f}, "
          f"最大: {max(desc_lengths)}, 最小: {min(desc_lengths)}")

    # 场景分布
    scenes = [s.get('scene') for s in samples if 'scene' in s]
    if scenes:
        scene_counts = Counter(scenes)
        print(f"场景分布: {dict(scene_counts)}")

    # 样本示例
    print("\n样本示例:")
    for i in range(min(3, len(samples))):
        print(f"  {i+1}. 图像: {samples[i]['image']}")
        print(f"     描述: {samples[i]['caption']}")
        print(f"     所属: {samples[i]['split']}")

if __name__ == '__main__':
    analyze_dataset('./annotations/cityflow-nl.json')