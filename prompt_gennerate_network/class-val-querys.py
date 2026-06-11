import json
import re
import os

save_dir = './class-evaluate'
val_querys_file = os.path.join('./checkpoints', 'val-querys.json')   # 车辆轨迹标注

with open(val_querys_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义分类规则函数
def classify_caption(caption):
    caption_lower = caption.lower()

    # 特殊天气条件（优先判断，因为可能包含其他关键词）
    weather_keywords = ['rain', 'snow', 'fog', 'night', 'dark', 'dirty', 'heavy rain', 'wet', 'low light', 'icy', 'freezing']
    for kw in weather_keywords:
        if kw in caption_lower:
            return 'weather'

    # 复杂交叉口
    intersection_keywords = ['intersection', 'turn left', 'turn right', 'cross', 'junction', 'roundabout', 'ramp', 'left turn', 'right turn', 'crosses', 'crossing', 'left onto', 'right onto']
    for kw in intersection_keywords:
        if kw in caption_lower:
            return 'intersection'

    # 拥堵环境
    congestion_keywords = ['follow', 'behind', 'ahead', 'queue', 'line of', 'traffic', 'busy', 'congestion', 'following', 'followed by', 'another vehicle', 'other car']
    for kw in congestion_keywords:
        if kw in caption_lower:
            return 'congestion'

    # 简单道路场景
    simple_keywords = ['straight', 'going straight', 'keep straight', 'runs down', 'drives down', 'runs on', 'drives on', 'forward', 'moving']
    for kw in simple_keywords:
        if kw in caption_lower:
            return 'simple'

    # 默认归类为简单道路场景
    return 'simple'

# 分类
classified = {
    'simple': [],
    'intersection': [],
    'congestion': [],
    'weather': []
}

for item in data:
    caption = item.get('caption', '')
    category = classify_caption(caption)
    classified[category].append(item)

# 保存为四个文件
with open(os.path.join(save_dir, 'simple_road_scenes.json'), 'w', encoding='utf-8') as f:
    json.dump(classified['simple'], f, indent=2, ensure_ascii=False)

with open(os.path.join(save_dir, 'complex_intersection.json'), 'w', encoding='utf-8') as f:
    json.dump(classified['intersection'], f, indent=2, ensure_ascii=False)

with open(os.path.join(save_dir, 'congestion_environment.json'), 'w', encoding='utf-8') as f:
    json.dump(classified['congestion'], f, indent=2, ensure_ascii=False)

with open(os.path.join(save_dir, 'special_weather.json'), 'w', encoding='utf-8') as f:
    json.dump(classified['weather'], f, indent=2, ensure_ascii=False)

# 打印统计信息
print(f"简单道路场景: {len(classified['simple'])} 条")
print(f"复杂交叉口: {len(classified['intersection'])} 条")
print(f"拥堵环境: {len(classified['congestion'])} 条")
print(f"特殊天气条件: {len(classified['weather'])} 条")