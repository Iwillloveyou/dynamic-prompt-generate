import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "test-queries.json": os.path.join(BASE_DIR, "test-queries.json"),
    "train-tracks.json": os.path.join(BASE_DIR, "train-tracks.json"),
    "test-tracks.json":  os.path.join(BASE_DIR, "test-tracks.json"),
}


def count_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)

def count():
    # 获取所有车辆 ID
    with open(track_ann_file, 'r') as f:
        tracks = json.load(f)
    split_file = os.path.join(Config.save_dir, 'track_split.pkl')   # 保存到 checkpoints 目录下
    train_track_ids, val_track_ids = get_or_create_track_split(
        Config.track_ann_file, split_file, train_ratio=0.8, seed=42
    )
    train_count = 0
    val_count = 0
    train_vehicles = 0
    val_vehicles = 0

    for track_id, info in tracks.items():
        frames = info.get('frames', [])
        num_images = len(frames)
        if track_id in train_track_ids:
            train_count += num_images
            train_vehicles += 1
        elif track_id in val_track_ids:
            val_count += num_images
            val_vehicles += 1
        else:
            # 可能有些 track 不在划分中，忽略
            pass

    # 4. 输出结果
    print("Dataset Image Count Summary")
    print(f"Training set:")
    print(f"  Vehicles: {train_vehicles}")
    print(f"  Images:   {train_count}")
    print()
    print(f"Validation set:")
    print(f"  Vehicles: {val_vehicles}")
    print(f"  Images:   {val_count}")
    print()
    print(f"Total vehicles: {train_vehicles + val_vehicles}")
    print(f"Total images:   {train_count + val_count}")

if __name__ == "__main__":
    print(f"{'文件':<25} {'条数':>8}")
    print("-" * 35)
    for name, path in FILES.items():
        count = count_json(path)
        print(f"{name:<25} {count:>8}")
