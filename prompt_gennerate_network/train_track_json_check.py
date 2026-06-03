import os
import json
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 加载标注
with open('train_tracks.json', 'r') as f:
    tracks = json.load(f)

# 保存图片的目录（自动创建）
output_dir = "./checkpoints/vis_results/"
os.makedirs(output_dir, exist_ok=True)

# 随机选择 3 个 track
sample_tracks = random.sample(list(tracks.keys()), 3)

def draw_box(draw, box, color='red', width=3):
    x, y, w, h = box

    # 计算右下角坐标
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # 画框（PIL 标准格式）
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

for track_id in sample_tracks:
    info = tracks[track_id]
    frames = info['frames']
    boxes = info['boxes']
    nl_list = info['nl']

    print(f"\nTrack ID: {track_id}")
    print(f"Descriptions: {nl_list}")

    # 随机选择 3 帧
    for idx in range(len(frames)):
    # for idx in random.sample(range(len(frames)), min(3, len(frames))):
        img_path = os.path.join('../../dataset/cityflow-nl/', frames[idx])
        box = boxes[idx]

        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        # 画框
        # draw.rectangle(fixed_box, outline='red', width=3)
        draw_box(draw, box, color='red', width=3)
        # 保存到本地（关键：不用 img.show()！）
        save_path = os.path.join(output_dir, f"track_{track_id}_frame_{idx}.jpg")
        img.save(save_path)
        print(f"✅ 已保存：{save_path}")
        # input("Press Enter to continue...")
