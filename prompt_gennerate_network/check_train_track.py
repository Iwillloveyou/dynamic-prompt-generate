import json
from collections import defaultdict

# ========== 配置路径（改成你自己的路径） ==========
json_path = "./train_tracks.json"
output_path = "./scene_info.txt"

# ========== 1. 加载 JSON 文件 ==========
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ 成功加载文件，文件大小：{len(json.dumps(data)) / 1024 / 1024:.2f} MB")

# ========== 2. 统计 track_id 数量 ==========
track_ids = list(data.keys())
total_tracks = len(track_ids)
print(f"\n📊 总 track_id 数量：{total_tracks}")

# ========== 3. 解析 frames，统计出现的场景 ==========
# 用集合自动去重
scenes = set()
scene_to_cameras = defaultdict(set)  # 场景 -> 摄像头集合

for track_id, track_info in data.items():
    frames = track_info.get("frames", [])
    for frame_path in frames:
        # frame_path 示例：./validation/S02/c006/img1/0000001.jpg
        parts = frame_path.strip().split("/")
        if len(parts) >= 4:
            # parts[-4] 是 S02，parts[-3] 是 c006
            scene = parts[-4]
            camera = parts[-3]
            scenes.add(scene)
            scene_to_cameras[scene].add(camera)

# ========== 4. 输出结果 ==========
print(f"\n🔍相关的场景（去重后）：")
for scene in sorted(scenes):
    cameras = sorted(scene_to_cameras[scene])
    print(f"  - 场景：{scene}，摄像头：{', '.join(cameras)}")

# ========== 5. 保存结果到文件 ==========
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"总 track_id 数量：{total_tracks}\n\n")
    f.write("相关场景与摄像头统计：\n")
    for scene in sorted(scenes):
        cameras = sorted(scene_to_cameras[scene])
        f.write(f"- 场景：{scene}，摄像头：{', '.join(cameras)}\n")

print(f"\n💾 结果已保存到：{output_path}")