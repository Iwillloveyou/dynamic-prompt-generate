# parse_annotations.py
import json
import os
from collections import defaultdict

def parse_cityflow_nl_annotations(data_dir, output_file):
    """
    解析CityFlow-NL标注文件，生成统一的训练格式
    """
    # 加载跟踪标注和查询标注
    with open(os.path.join(data_dir, 'train-tracks.json'), 'r') as f:
        tracks = json.load(f)

    with open(os.path.join(data_dir, 'test-queries.json'), 'r') as f:
        queries = json.load(f)

    # 构建图像-描述对
    samples = []

    # 处理训练集跟踪标注（每辆车可能有多个帧的描述）
    for track_id, track_info in tracks.items():
        vehicle_id = track_info.get('vehicle_id')
        nl_descriptions = track_info.get('nl', [])

        # 遍历该车辆出现的所有帧
        for frame_info in track_info.get('frames', []):
            scene = frame_info['scene']
            camera = frame_info['camera']
            frame_num = frame_info['frame']

            image_path = f"{scene}/{camera}/frame_{frame_num:06d}.jpg"

            # 每条描述作为一个独立样本
            for desc in nl_descriptions:
                samples.append({
                    'image': image_path,
                    'caption': desc,
                    'split': 'train',
                    'vehicle_id': vehicle_id,
                    'track_id': track_id,
                    'scene': scene,
                    'camera': camera,
                    'frame': frame_num
                })

    # 处理测试集查询
    for query in queries:
        image_path = query.get('image_path')  # 可能需要根据实际格式调整
        nl = query.get('nl', '')
        samples.append({
            'image': image_path,
            'caption': nl,
            'split': 'test',
            'query_id': query.get('id')
        })

    # 保存统一格式的标注文件
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"总样本数: {len(samples)}")
    print(f"训练集样本: {len([s for s in samples if s['split']=='train'])}")
    print(f"测试集样本: {len([s for s in samples if s['split']=='test'])}")

    return samples

if __name__ == '__main__':
    parse_cityflow_nl_annotations(
        data_dir='./data',
        output_file='./annotations/cityflow-nl.json'
    )