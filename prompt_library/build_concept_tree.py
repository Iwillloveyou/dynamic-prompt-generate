#!/usr/bin/env python3
"""
概念树构建程序（修正版）
输入: openadd.json (包含概念层次结构)
输出: concept_tree.json (树形结构)
功能: 递归处理所有概念，包括 children 数组中的子节点
"""

import json
from typing import Dict, List, Any, Optional, Set


class TreeNode:
    """树节点类"""
    def __init__(self, name: str, parent: Optional[str] = None):
        self.name = name
        self.parent = parent
        self.children = []

    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典格式"""
        return {
            "name": self.name,
            "parent": self.parent,
            "children": [child.to_dict() for child in self.children]
        }

    def __repr__(self):
        return f"TreeNode(name={self.name}, parent={self.parent}, children={len(self.children)})"


def load_concepts(json_path: str) -> Dict[str, Any]:
    """加载概念JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_all_concepts(concept_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    递归收集所有概念，包括 children 数组中的子节点
    返回: {概念名: 概念信息} 的字典
    """
    all_concepts = {}

    def process_concept(concept_name: str, parent_name: Optional[str] = None):
        """递归处理单个概念及其子节点"""
        if concept_name in all_concepts:
            return  # 已经处理过，避免循环

        # 获取概念信息
        concept_info = concept_data.get(concept_name, {})

        # 记录当前概念
        all_concepts[concept_name] = {
            "name": concept_name,
            "parent": parent_name,
            "children": concept_info.get("children", [])
        }

        # 递归处理子节点
        for child in concept_info.get("children", []):
            process_concept(child, concept_name)

    # 从所有顶层概念开始处理
    for concept_name in concept_data.keys():
        # 检查是否已经有父节点（避免重复处理）
        if concept_name not in all_concepts:
            process_concept(concept_name, None)

    return all_concepts


def build_tree(concept_data: Dict[str, Any]) -> TreeNode:
    """
    根据概念数据构建树形结构
    返回根节点 TreeNode
    """
    # 第一步：收集所有概念（包括children中的）
    all_concepts = collect_all_concepts(concept_data)

    print(f"📊 收集到 {len(all_concepts)} 个概念")

    # 第二步：创建所有节点字典
    nodes: Dict[str, TreeNode] = {}

    for concept_name, concept_info in all_concepts.items():
        parent = concept_info.get("parent")
        nodes[concept_name] = TreeNode(concept_name, parent)

    # 第三步：建立父子关系
    root_node = None
    root_candidates = []

    for concept_name, node in nodes.items():
        if node.parent is None:
            # 这是根节点候选
            root_candidates.append(node)
        else:
            # 将当前节点添加到父节点的children中
            if node.parent in nodes:
                nodes[node.parent].children.append(node)
            else:
                print(f"⚠️ 警告: 概念 '{concept_name}' 的父节点 '{node.parent}' 不存在")

    # 第四步：确定根节点
    if len(root_candidates) == 1:
        root_node = root_candidates[0]
        if root_node.name != "OperationalDesignDomain":
            print(f"⚠️ 检测到根节点是 '{root_node.name}'，但期望是 'OperationalDesignDomain'")
    else:
        # 如果有多个根节点，尝试找到 OperationalDesignDomain
        odd_candidates = [n for n in root_candidates if n.name == "OperationalDesignDomain"]
        if odd_candidates:
            root_node = odd_candidates[0]
            # 将其它根节点作为子节点添加到 OperationalDesignDomain 下
            for candidate in root_candidates:
                if candidate != root_node:
                    print(f"🔗 将根节点 '{candidate.name}' 作为子节点添加到 OperationalDesignDomain 下")
                    root_node.children.append(candidate)
                    candidate.parent = "OperationalDesignDomain"
        else:
            print(f"❌ 无法确定根节点，找到 {len(root_candidates)} 个根节点: {[n.name for n in root_candidates]}")
            # 默认取第一个
            root_node = root_candidates[0]

    return root_node


def print_tree(node: TreeNode, level: int = 0, prefix: str = "", is_last: bool = True) -> None:
    """
    打印树形结构（用于调试）
    """
    if level == 0:
        print(node.name)
    else:
        connector = "└── " if is_last else "├── "
        print(prefix + connector + node.name)
        prefix += "    " if is_last else "│   "

    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        print_tree(child, level + 1, prefix, is_last_child)


def save_tree_to_json(root_node: TreeNode, output_path: str) -> None:
    """将树保存为JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(root_node.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"✅ 树结构已保存至: {output_path}")


def get_tree_statistics(root_node: TreeNode) -> Dict[str, Any]:
    """统计树的基本信息"""
    def count_nodes(node: TreeNode) -> int:
        return 1 + sum(count_nodes(child) for child in node.children)

    def get_max_depth(node: TreeNode, current_depth: int = 0) -> int:
        if not node.children:
            return current_depth
        return max(get_max_depth(child, current_depth + 1) for child in node.children)

    def count_leaves(node: TreeNode) -> int:
        if not node.children:
            return 1
        return sum(count_leaves(child) for child in node.children)

    total_nodes = count_nodes(root_node)
    max_depth = get_max_depth(root_node)
    leaf_count = count_leaves(root_node)

    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "leaf_count": leaf_count,
        "root_name": root_node.name
    }


def validate_tree_structure(root_node: TreeNode, all_concepts: Dict) -> bool:
    """
    验证树结构是否正确
    """
    visited = set()

    def dfs(node: TreeNode):
        if node.name in visited:
            print(f"❌ 检测到循环引用: {node.name}")
            return False
        visited.add(node.name)

        # 验证所有子节点都存在
        for child in node.children:
            if child.name not in all_concepts:
                print(f"❌ 子节点 {child.name} 不在概念集合中")
                return False
            if not dfs(child):
                return False
        return True

    return dfs(root_node)


def main():
    """主函数"""
    input_file = "openadd.json"
    output_file = "result/concept_tree.json"

    print(f"📖 读取概念文件: {input_file}")

    # 1. 加载数据
    concept_data = load_concepts(input_file)
    print(f"📊 JSON文件包含 {len(concept_data)} 个顶层概念")

    # 2. 收集所有概念
    all_concepts = collect_all_concepts(concept_data)

    # 3. 构建树
    print("🌳 构建树形结构...")
    root_node = build_tree(concept_data)

    if not root_node:
        print("❌ 无法找到根节点")
        return

    # 4. 验证树结构
    if validate_tree_structure(root_node, all_concepts):
        print("✅ 树结构验证通过")
    else:
        print("⚠️ 树结构验证发现问题")

    # 5. 打印树结构
    print("\n" + "="*50)
    print("🌲 概念树结构:")
    print("="*50)
    print_tree(root_node)

    # 6. 统计信息
    stats = get_tree_statistics(root_node)
    print("\n" + "="*50)
    print("📊 树统计信息:")
    print("="*50)
    print(f"总节点数: {stats['total_nodes']}")
    print(f"最大深度: {stats['max_depth']}")
    print(f"叶子节点数: {stats['leaf_count']}")
    print(f"根节点: {stats['root_name']}")

    # 7. 保存为JSON
    save_tree_to_json(root_node, output_file)

    # 8. 可选：也保存为更简洁的父子关系列表
    simplified_output = "concept_tree_simple.json"
    with open(simplified_output, 'w', encoding='utf-8') as f:
        # 生成简化的父子关系列表
        simple_tree = []
        def collect_nodes(node: TreeNode, level: int):
            simple_tree.append({
                "name": node.name,
                "parent": node.parent,
                "level": level,
                "has_children": len(node.children) > 0,
                "children_count": len(node.children),
                "children_names": [c.name for c in node.children]
            })
            for child in node.children:
                collect_nodes(child, level + 1)

        collect_nodes(root_node, 0)
        json.dump(simple_tree, f, indent=2, ensure_ascii=False)

    print(f"✅ 简化列表已保存至: {simplified_output}")

    # 9. 输出一些统计信息
    print("\n📈 概念分布:")
    leaf_nodes = [node for node in simple_tree if not node["has_children"]]
    internal_nodes = [node for node in simple_tree if node["has_children"]]
    print(f"  内部节点数: {len(internal_nodes)}")
    print(f"  叶子节点数: {len(leaf_nodes)}")

    # 找出最深的路径
    max_level_node = max(simple_tree, key=lambda x: x["level"])
    print(f"  最大深度: {max_level_node['level']} (概念: {max_level_node['name']})")

    print("\n🎉 完成!")


if __name__ == "__main__":
    main()