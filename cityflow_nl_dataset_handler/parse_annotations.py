# parse_annotations.py
import json
import os
from collections import defaultdict

#解析train-tracks.json 将其构建成图像-描述对

def parse_cityflow_nl_annotations(data_dir, output_file):
    """
    解析CityFlow-NL标注文件，生成统一的训练格式
    """
    # 加载跟踪标注和查询标注
    with open(os.path.join(data_dir, 'train-tracks.json'), 'r') as f:
        tracks = json.load(f)

    # with open(os.path.join(data_dir, 'test-queries.json'), 'r') as f:
    #     queries = json.load(f)

    # 构建图像-描述对
    samples = []
    # data_root = "/root/autodl-tmp/cityflow-nl-reproduce/data/"

    # 测试模式：只处理第一个概念
    test_mode = True  # 设为 False 时处理全部概念
    if test_mode:
        # 取字典的第一个键值对，转成新字典（保持原数据结构）
        first_key = next(iter(tracks.keys()))  # 获取第一个键
        tracks_to_process = {first_key: tracks[first_key]}
    else:
        # 处理全部
        tracks_to_process = tracks

    # 处理训练集跟踪标注（每辆车可能有多个帧的描述）
    for track_id, track_info in tracks_to_process.items():
        nl_descriptions = track_info.get('nl', [])

        # 遍历该车辆出现的所有帧
        for frame_info in track_info.get('frames', []):
            # scene = frame_info['scene']
            # camera = frame_info['camera']
            # frame_num = frame_info['frame']

            image_path = frame_info[2:]

            # 每条描述作为一个独立样本
            for desc in nl_descriptions:
                samples.append({
                    'image': image_path,
                    'caption': desc,
                    'split': 'train',
                    'track_id': track_id
                })

    # 处理测试集查询
    # for query in queries:
    #     image_path = query.get('image_path')  # 可能需要根据实际格式调整
    #     nl = query.get('nl', '')
    #     samples.append({
    #         'image': image_path,
    #         'caption': nl,
    #         'split': 'test',
    #         'query_id': query.get('id')
    #     })

    # 保存统一格式的标注文件
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"总样本数: {len(samples)}")
    print(f"训练集样本: {len([s for s in samples if s['split']=='train'])}")
    print(f"测试集样本: {len([s for s in samples if s['split']=='test'])}")

    return samples

if __name__ == '__main__':
    output_dir = "./annotations/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    parse_cityflow_nl_annotations(
        data_dir='./',
        output_file= output_dir + 'cityflow-nl.json'
    )