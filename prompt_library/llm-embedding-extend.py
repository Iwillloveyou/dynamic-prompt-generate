"""
概念语义扩展与向量库构建脚本
输入: openadd.json (包含概念层次结构)
输出: concept_vectors.npy (向量数组), concept_names.json (概念名称列表)
依赖: pip install openai torch transformers numpy tqdm
"""

import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm
import openai  # 或使用其他SDK，此处以OpenAI为例
import os
import time
from typing import List, Dict, Any

# ==================== 配置区域 ====================
OPENAI_API_KEY = "your-api-key"  # 请替换为实际API密钥
OPENAI_API_BASE = "https://api.openai.com/v1"  # 如使用代理或兼容接口可修改
MODEL_NAME = "gpt-4"  # 或 "deepseek-chat", "doubao-pro" 等，需相应调整客户端
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # HuggingFace上的CLIP模型
JSON_PATH = "openadd.json"
OUTPUT_VECTORS = "concept_vectors.npy"
OUTPUT_NAMES = "concept_names.json"
NUM_DESCRIPTIONS_PER_CONCEPT = 8  # 每个概念生成多少条描述
TEMPERATURE = 0.8  # LLM生成多样性
# =================================================

# 初始化OpenAI客户端（示例，其他模型请修改）
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)

# ==================== 辅助函数：属性推断 ====================
def get_attributes_for_concept(concept: str, category: str) -> List[str]:
    """
    根据概念名称和类别返回建议的属性维度列表。
    这里使用简单的规则映射，可根据实际需要扩展。
    """
    # 基于类别的基本属性模板
    category_attributes = {
        "Weather": ["强度", "时间段", "能见度", "伴随现象"],
        "Illumination": ["光照强度", "光源类型", "时间段", "阴影"],
        "Precipitation": ["降水类型", "强度", "持续时间", "影响"],
        "Fog": ["浓度", "能见度", "时间段", "高度"],
        "RoadType": ["道路类型", "车道数", "限速", "隔离带"],
        "RoadSurface": ["表面材质", "湿滑程度", "损坏情况", "摩擦系数"],
        "TrafficParticipants": ["类型", "数量", "运动状态", "位置"],
        "Vehicle": ["车辆类型", "速度", "灯光状态", "行驶意图"],
        "VulnerableRoadUser": ["类型", "行为", "可见性", "保护装备"],
        "DrivingDynamics": ["速度范围", "加速度", "方向变化", "紧急程度"],
        "Infrastructure": ["类型", "状态", "可见性", "功能性"],
        "TemporalConditions": ["具体时间", "季节", "节假日", "特殊事件"],
        "LegalRegulatoryConditions": ["规则类型", "限制值", "适用区域", "执行力度"],
        "VehicleCapabilities": ["能力类型", "性能参数", "可用性", "可靠性"],
        "Perception": ["感知对象", "距离", "准确率", "环境条件"],
        "Map": ["地图类型", "精度", "更新频率", "覆盖区域"],
        "Traffic": ["流量状态", "密度", "组成", "规则符合性"],
        "Road": ["道路特征", "几何", "标线", "附属设施"],
        "Environment": ["环境因素", "综合影响"],
    }
    # 默认属性
    default_attrs = ["强度", "场景类型", "时间段", "影响程度"]

    # 如果category在映射中，返回对应属性；否则返回默认
    attrs = category_attributes.get(category, default_attrs)

    # 可根据特定概念微调
    special_rules = {
        "Clear": ["云量", "能见度", "光照条件", "降水"],
        "Cloudy": ["云量", "光照", "降水可能"],
        "Rain": ["降雨强度", "持续时间", "路面湿滑", "能见度"],
        "Snow": ["降雪强度", "积雪厚度", "路面结冰", "能见度"],
        "Daylight": ["时间", "太阳高度", "光照强度", "阴影"],
        "Night": ["月光", "人工照明", "可见度", "时段"],
        "Dry": ["路面干燥程度", "温度", "灰尘"],
        "Wet": ["积水程度", "潮湿", "反光", "滑移风险"],
        "Icy": ["冰层厚度", "黑冰", "温度", "摩擦系数"],
    }
    if concept in special_rules:
        attrs = special_rules[concept]
    return attrs

# ==================== 生成描述的提示词模板 ====================
def build_prompt(concept: str, category: str, attributes: List[str]) -> str:
    """
    构建发送给LLM的提示词，要求生成多样化的自动驾驶场景描述。
    """
    attr_str = "、".join(attributes)
    prompt = f"""你是一位资深的自动驾驶场景工程师，负责为自动驾驶数据集生成高质量的图文对训练数据。

【核心概念】：{concept}
【概念类别】：{category}
【关键属性】：请考虑以下属性维度的变化：{attr_str}

【任务要求】
请基于上述核心概念，生成{NUM_DESCRIPTIONS_PER_CONCEPT}条关于自动驾驶场景的自然语言描述。这些描述将用于训练图文检索模型，要求如下：
1. 多样性：覆盖不同的属性状态（如强度变化、不同时间段、不同道路类型、不同交通密度等）。
2. 真实性：描述必须是真实的驾驶场景视角（第一人称或车载摄像头视角），像是车载摄像头会捕捉到的画面。
3. 专业性：使用专业但自然的驾驶描述语言。
4. 输出格式：请以JSON列表格式输出，不要有其他解释性文字。每个描述是一个字符串。

【生成示例】（针对“RedLight”概念）
- "车辆在城市道路路口等待红灯，前方有数辆轿车和一辆公交车，交通信号灯为红色。"
- "夜间行驶至十字路口，红灯亮起，车辆逐渐减速停止，左侧车道有一辆SUV也在等待。"

请开始生成针对“{concept}”的描述：
"""
    return prompt

def call_llm(prompt: str) -> List[str]:
    """
    调用LLM API生成描述，返回字符串列表。
    此处以OpenAI API为例，若使用其他模型请修改。
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=500,
            n=1,
            stop=None
        )
        content = response.choices[0].message.content.strip()
        # 尝试解析JSON列表
        try:
            descriptions = json.loads(content)
            if isinstance(descriptions, list):
                return descriptions
            else:
                print(f"返回格式不是列表: {content[:100]}")
                return []
        except json.JSONDecodeError:
            # 如果返回的不是合法JSON，尝试按行分割或直接作为单个描述
            lines = [line.strip("- ").strip() for line in content.split("\n") if line.strip()]
            return lines
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return []

# ==================== CLIP编码函数 ====================
def encode_texts(texts: List[str]) -> np.ndarray:
    """
    使用CLIP模型编码文本列表，返回归一化后的向量数组 (n, dim)。
    """
    if not texts:
        return None
    inputs = clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2归一化
    return embeddings.cpu().numpy()

# ==================== 主流程 ====================
def main():
    # 1. 加载JSON概念树
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        concept_data = json.load(f)

    # 提取所有概念及其父类别
    concepts = []
    for key, value in concept_data.items():
        concept = key
        parent = value.get("parent", None)
        # 如果parent为null，根节点，类别设为"Root"
        category = parent if parent else "OperationalDesignDomain"
        concepts.append({
            "name": concept,
            "category": category,
            "has_children": len(value.get("children", [])) > 0
        })

    print(f"共加载 {len(concepts)} 个概念。")

    # 2. 为每个概念生成描述并编码
    concept_vectors = {}
    failed_concepts = []

    for item in tqdm(concepts, desc="处理概念"):
        concept = item["name"]
        category = item["category"]
        # 推断属性
        attributes = get_attributes_for_concept(concept, category)

        # 构建提示词
        prompt = build_prompt(concept, category, attributes)

        # 调用LLM生成描述
        descriptions = call_llm(prompt)
        if not descriptions:
            print(f"警告：概念 {concept} 未生成描述，跳过。")
            failed_concepts.append(concept)
            continue

        # 取前NUM_DESCRIPTIONS_PER_CONCEPT条（或全部）
        descriptions = descriptions[:NUM_DESCRIPTIONS_PER_CONCEPT]

        # 编码
        vecs = encode_texts(descriptions)
        if vecs is None or len(vecs) == 0:
            print(f"警告：概念 {concept} 编码失败。")
            failed_concepts.append(concept)
            continue

        # 取均值作为概念向量
        concept_vector = vecs.mean(axis=0)
        # 再次归一化（可选）
        concept_vector = concept_vector / np.linalg.norm(concept_vector)
        concept_vectors[concept] = concept_vector

        # 避免API限流，适当延时
        time.sleep(0.5)

    # 3. 保存结果
    # 将概念向量按顺序保存为数组，同时保存名称列表
    concept_names = list(concept_vectors.keys())
    vectors = np.array([concept_vectors[name] for name in concept_names])

    np.save(OUTPUT_VECTORS, vectors)
    with open(OUTPUT_NAMES, "w", encoding="utf-8") as f:
        json.dump(concept_names, f, indent=2)

    print(f"成功处理 {len(concept_names)} 个概念，失败 {len(failed_concepts)} 个。")
    if failed_concepts:
        print("失败概念:", failed_concepts)
    print(f"向量维度: {vectors.shape[1]}")
    print(f"向量库已保存至 {OUTPUT_VECTORS} 和 {OUTPUT_NAMES}")

if __name__ == "__main__":
    main()