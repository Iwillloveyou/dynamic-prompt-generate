import json

def extract_concepts_two_pass(concept_data):
    """
    两遍遍历提取所有概念：
    第一遍：收集所有节点
    第二遍：建立父子关系
    """
    # 第一遍：收集所有节点
    all_nodes = {}

    # 首先添加所有顶层节点
    for key, value in concept_data.items():
        if key not in all_nodes:
            all_nodes[key] = {
                "name": key,
                "parent": None,
                "category": "OperationalDesignDomain",
                "has_children": len(value.get("children", [])) > 0
            }

        # 添加子节点
        for child in value.get("children", []):
            if child not in all_nodes:
                child_value = concept_data.get(child, {})
                all_nodes[child] = {
                    "name": child,
                    "parent": key,  # 先临时设置父节点
                    "category": key,  # 类别就是父节点
                    "has_children": len(child_value.get("children", [])) > 0
                }

    return all_nodes

JSON_PATH = "openadd.json"

# def main():
# 1. 加载 JSON 概念树
print(f"加载概念文件: {JSON_PATH}")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    concept_data = json.load(f)

concepts_dict = extract_concepts_two_pass(concept_data)

print("提取的概念列表：")
for name, info in sorted(concepts_dict.items()):
    print(f"  {name}: 父={info['parent']}, 类别={info['category']}, 有子节点={info['has_children']}")

# 提取所有概念及其父类别
concepts = list(concepts_dict.values())

print(f"共加载 {len(concepts)} 个概念。")