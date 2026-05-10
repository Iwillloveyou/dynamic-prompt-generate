import requests
import time
import json

"""
获取鉴智odd标签并构建odd.json
"""

# ===================== 配置项（请根据你的实际情况修改）=====================
BASE_URL = "http://data.phigent.net/api/label-api/label/tree/query"  # 替换为真实接口地址
AUTH_TOKEN = "Bearer 你的授权token"  # 替换为你的 authorization 凭证
REQUEST_INTERVAL = 0.2  # 请求间隔 200ms
ROOT_KEY = "OperationalDesignDomain"  # 根节点固定key
SAVE_FILE = "config/phigent/phigent_full_openodd.json"  # 保存的文件名
# =========================================================================

# 最终存储结果的字典
result_tree = {}

def request_label_tree(parent_id: str) -> list:
    """
    请求标签树接口，返回 list 数据
    :param parent_id: 父节点ID
    :return: 接口返回的 list 数组
    """
    headers = {
        "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo3MjY3LCJ1aWQiOiJrb2c0dXJxaCIsInVzZXJuYW1lIjoiamlhbnBlbmcueXVhbiIsIm5pY2siOiLoooHlu7rmo5oiLCJyb2xlcyI6WzEyLDE0LDE4LDMyLDM3LDUxLDUyLDU0LDYxXSwicmV4cCI6MTc3NzcwMzEwOSwiaXNzIjoiYXV0aC1hcGkiLCJleHAiOjE3NzU1NDMxMDksImlhdCI6MTc3NTExMTEwOX0.61HOjFKnLXuPoiNd1ZAOMqL_l89dRxid05_gBep0bEY",
        "Content-Type": "application/json"
    }
    payload = {
        "parent_id": parent_id,
        "with_formatted_attrs": True,
        "with_attrs": True,
        "with_user": True,
        "with_child_cnt": True,
        "clazz": 3
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()  # 抛出HTTP异常
        data = response.json()

        # 校验接口返回码
        if data.get("code") != 0:
            print(f"接口请求失败，parent_id={parent_id}，错误信息：{data}")
            return []

        return data.get("data", {}).get("list", [])

    except Exception as e:
        print(f"请求异常 parent_id={parent_id}：{str(e)}")
        return []


def build_node(parent_key: str, current_parent_id: str):
    """
    递归构建节点结构
    :param parent_key: 父节点的 key
    :param current_parent_id: 当前请求的 parent_id
    """
    # 1. 请求接口获取子节点列表
    node_list = request_label_tree(current_parent_id)
    if not node_list:
        return

    # 2. 提取当前节点的所有 key，存入父节点的 children
    child_keys = [node["key"] for node in node_list]
    result_tree[parent_key]["children"] = child_keys

    # 3. 遍历每个子节点，创建节点结构 + 递归处理子节点
    for node in node_list:
        node_key = node["key"]
        has_child = node.get("has_child", False)
        node_id = node["id"]

        # 创建当前节点结构
        result_tree[node_key] = {
            "parent": parent_key,
            "children": []
        }

        # 如果有子节点，递归请求（间隔200ms）
        if has_child:
            time.sleep(REQUEST_INTERVAL)
            build_node(parent_key=node_key, current_parent_id=node_id)


if __name__ == "__main__":
    # 初始化根节点
    result_tree[ROOT_KEY] = {
        "parent": None,
        "children": []
    }

    print("开始构建标签树结构...")
    # 从根节点 parent_id=0 开始递归构建
    build_node(parent_key=ROOT_KEY, current_parent_id="0")

    # ===================== 保存到 JSON 文件 =====================
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(result_tree, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 构建完成！结果已保存到：{SAVE_FILE}")
    print("文件预览：")
    print(json.dumps(result_tree, ensure_ascii=False, indent=2))