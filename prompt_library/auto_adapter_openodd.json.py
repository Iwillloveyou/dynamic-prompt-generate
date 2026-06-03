import json
import csv
import re

# ==================== 1. 加载数据 ====================
with open('openodd_desc.json', 'r', encoding='utf-8') as f:
    openodd = json.load(f)

keywords = []
with open('high_freq_words_single.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        word = row['word'].strip().lower()
        if word:
            keywords.append(word)

# ==================== 2. 定义匹配规则 ====================
# 规则：根据关键词的语义，返回 (父节点路径, 描述模板)
# 为了覆盖全部500个词，使用分类匹配+默认路径

def get_parent_and_desc(word):
    # 颜色词
    colors = {'white','black','gray','grey','silver','red','blue','green','yellow','purple','brown','orange','beige','tan','maroon','burgundy','gold','champagne','navy','metallic','dark','light'}
    if word in colors:
        return ('Traffic/TrafficParticipants/Vehicle/Car', f"A vehicle with {word} exterior paint.")

    # 车型词
    car_types = {
        'suv': 'Sport utility vehicle with raised ground clearance and cargo capacity.',
        'sedan': 'Passenger car with a three-box configuration: engine, passenger, and cargo compartments.',
        'hatchback': 'Car with a rear door that swings upward, combining passenger and cargo space.',
        'wagon': 'Station wagon with extended cargo area and rear liftgate.',
        'coupe': 'Two-door car with a sloping roofline.',
        'convertible': 'Car with a retractable roof.',
        'minivan': 'Multi-purpose vehicle designed for family transport.',
        'mpv': 'Multi-purpose vehicle with flexible seating.',
        'crossover': 'Vehicle that combines features of a SUV and a car.',
        'pickup': 'Light truck with an open cargo bed.',
        'truck': 'Large motor vehicle for carrying heavy cargo.',
        'van': 'Box-shaped vehicle for transporting goods or passengers.',
        'bus': 'Large passenger-carrying vehicle.',
        'motorcycle': 'Two-wheeled motorized vehicle.',
        'bicycle': 'Non-motorized two-wheeled vehicle.',
        'taxi': 'Vehicle for hire with a driver.',
        'police': 'Emergency vehicle used by law enforcement.',
        'ambulance': 'Emergency vehicle for medical transport.',
        'firetruck': 'Emergency vehicle for firefighting.',
    }
    if word in car_types:
        # 判断是否为卡车/货车
        if word in ('truck','pickup','van'):
            return ('Traffic/TrafficParticipants/Vehicle/Truck', car_types[word])
        elif word in ('bus','motorcycle','bicycle'):
            return ('Traffic/TrafficParticipants/Vehicle/' + word.capitalize(), car_types[word])
        else:
            return ('Traffic/TrafficParticipants/Vehicle/Car', car_types[word])

    # 道路几何
    road_geo = {'curve','curved','curvy','bend','hill','slope','steep','incline','decline'}
    if word in road_geo:
        if word in ('curve','curved','curvy','bend'):
            return ('Road/RoadGeometry', 'A bend in the road alignment.')
        elif word in ('hill','slope','steep','incline','decline'):
            return ('Road/RoadGeometry', 'An inclined section of roadway.')

    # 路面状态
    surface = {'dry','wet','icy','snowy','muddy','pothole','potholes','rough','smooth'}
    if word in surface:
        return ('Road/RoadSurface', f"Road surface condition: {word}.")

    # 交通参与者（行人等）
    vulnerable = {'pedestrian','pedestrians','cyclist','wheelchair','animal','dog','cat','deer'}
    if word in vulnerable:
        if word.startswith('pedestrian'):
            return ('Traffic/TrafficParticipants/VulnerableRoadUser/Pedestrian', 'Person traveling on foot.')
        elif word == 'cyclist':
            return ('Traffic/TrafficParticipants/VulnerableRoadUser/Cyclist', 'Person riding a bicycle.')
        else:
            return ('Traffic/TrafficParticipants/Animal', f"A {word} on or near the roadway.")

    # 驾驶动作
    actions = {
        'straight':'Moving forward without turning.',
        'turn':'Changing direction from a straight path.',
        'turns':'Changing direction from a straight path.',
        'turning':'In the process of changing direction.',
        'right':'A turn to the right side.',
        'left':'A turn to the left side.',
        'stop':'Ceasing movement.',
        'stops':'Ceasing movement.',
        'stopped':'Vehicle that is not moving.',
        'park':'Stationary in a designated area.',
        'parked':'Vehicle stationary in a designated area.',
        'parking':'Act of parking a vehicle.',
        'go':'Proceeding forward.',
        'goes':'Moves from one location to another.',
        'going':'Progressing in a specific direction.',
        'drive':'Operating a vehicle.',
        'drives':'Operating a vehicle.',
        'driving':'Operating a vehicle.',
        'follow':'Driving behind another vehicle.',
        'follows':'Driving behind another vehicle.',
        'followed':'Driving behind another vehicle.',
        'following':'Maintaining a following distance.',
        'behind':'Positioned at the rear of another vehicle.',
        'overtake':'Passing another vehicle.',
        'overtaking':'Passing another vehicle.',
        'merge':'Joining traffic from another lane.',
        'merging':'Joining traffic.',
        'yield':'Giving right-of-way.',
        'yielding':'Giving right-of-way.',
        'accelerate':'Increase speed.',
        'accelerating':'Increasing speed.',
        'decelerate':'Decrease speed.',
        'brake':'Apply brakes to reduce speed.',
        'braking':'Applying brakes.',
    }
    if word in actions:
        return ('DrivingDynamics/LateralMovement' if word in ('straight','turn','turns','turning','right','left') else
                'DrivingDynamics/Stopping' if word in ('stop','stops','stopped','park','parked','parking') else
                'DrivingDynamics/Speed' if word in ('go','goes','going','drive','drives','driving','accelerate','accelerating','decelerate') else
                'Traffic/TrafficFlow', actions[word])

    # 基础设施
    infra = {'traffic light','traffic signal','stoplight','stop sign','crosswalk','sidewalk','guardrail','barrier','railroad crossing'}
    if word in infra:
        if 'light' in word or 'signal' in word:
            return ('Infrastructure/TrafficLight', 'Signal device controlling vehicle and pedestrian flow.')
        elif 'sign' in word:
            return ('Infrastructure/TrafficSign', 'Visual sign conveying traffic information.')
        else:
            return ('Infrastructure/' + word.replace(' ','').capitalize(), f"Infrastructure element: {word}.")

    # 默认：放到对应的宽泛类别中（道路、交通等）
    # 根据词频或语义猜测
    return ('OperationalDesignDomain', f"A driving scenario element described as '{word}'.")

# ==================== 3. 辅助函数 ====================
def get_node_by_path(node, path_parts):
    if not path_parts:
        return node
    key = path_parts[0]
    if key in node:
        return get_node_by_path(node[key], path_parts[1:])
    return None

# ==================== 4. 构建扩展 ====================
existing_nodes = set(node.lower() for node in openodd.keys())
new_nodes = {}
parent_children_updates = {}

for kw in keywords:
    # 标准化节点名称：首字母大写，其余小写（保留原词空格可处理，但建议转为驼峰）
    node_name = kw[0].upper() + kw[1:] if len(kw)>1 else kw.upper()
    # 避免因大小写冲突
    if node_name.lower() in existing_nodes:
        continue
    path_str, desc = get_parent_and_desc(kw)
    path_parts = path_str.split('/')
    parent_obj = get_node_by_path(openodd, path_parts)
    if parent_obj is None:
        # 尝试寻找父节点的直接名称（如果路径不存在）
        if len(path_parts)==1:
            parent_name = path_parts[0]
            if parent_name in openodd:
                parent_obj = openodd[parent_name]
        if parent_obj is None:
            print(f"警告: 未找到父节点 {path_str} 用于关键词 {kw}")
            continue
    parent_name = path_parts[-1] if path_parts else 'OperationalDesignDomain'
    new_nodes[node_name] = {
        "parent": parent_name,
        "children": [],
        "desc": desc
    }
    if parent_name not in parent_children_updates:
        parent_children_updates[parent_name] = []
    parent_children_updates[parent_name].append(node_name)

# 添加新节点到 openodd
openodd.update(new_nodes)

# 更新父节点的 children
for pname, child_list in parent_children_updates.items():
    if pname in openodd:
        if "children" not in openodd[pname]:
            openodd[pname]["children"] = []
        existing = set(openodd[pname]["children"])
        for ch in child_list:
            if ch not in existing:
                openodd[pname]["children"].append(ch)

# 输出
with open('openodd_extended.json', 'w', encoding='utf-8') as f:
    json.dump(openodd, f, indent=2, ensure_ascii=False)

print(f"扩展完成，新增节点数：{len(new_nodes)}")