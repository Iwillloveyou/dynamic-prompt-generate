"""
概念语义扩展与向量库构建脚本 (豆包模型版本)
输入: openadd.json (包含概念层次结构)
输出: concept_vectors.npy (向量数组), concept_names.json (概念名称列表)
依赖: pip install openai torch transformers numpy tqdm python-dotenv tenacity
"""

import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm
import openai
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==================== 配置区域 ====================
# 加载环境变量（推荐将 API Key 存储在 .env 文件中）
load_dotenv()
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "3f948467-3cc4-4d7c-8697-eed5c97f8e17")  # 替换为你的豆包API Key

# 豆包 API 配置（火山方舟地址）[citation:1]
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 以官方文档为准
# 推荐模型 [citation:2]
MODEL_NAME = "deepseek-v3-2-251201"  # 深度思考模型
# 备选模型: "doubao-1.5-pro-32k" (通用文本生成)

# CLIP 模型配置
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# 文件路径
JSON_PATH = "openodd_desc.json"
RESULT_SAVE_DIR = "./result/"
OUTPUT_VECTORS = RESULT_SAVE_DIR + "concept_vectors.npy"
CONCEPT_EXTEND_OUTPUT_NAMES = RESULT_SAVE_DIR + "concept_extend.json"
CONCEPT_EXTEND_EMBEDDING_OUTPUT_NAMES= RESULT_SAVE_DIR + "concept_extend.embeddings.npz"

# 生成参数
NUM_DESCRIPTIONS_PER_CONCEPT = 8  # 每个概念生成多少条描述
TEMPERATURE = 0.8  # 控制多样性 (0.0-1.0) [citation:3]
MAX_TOKENS = 500  # 最大输出token数

# 重试配置
MAX_RETRIES = 3
RETRY_MIN_WAIT = 4
RETRY_MAX_WAIT = 10

def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    加载 JSON 配置文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件不存在: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: {file_path}, {e}")
        return {}

# 配置文件路径（可以根据需要修改）
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '')
CATEGORY_ATTRS_PATH = os.path.join(CONFIG_DIR, 'category_attributes.json')
SPECIAL_RULES_PATH = os.path.join(CONFIG_DIR, 'special_rules.json')

# 全局加载配置（只在模块加载时执行一次）
CATEGORY_ATTRIBUTES = load_json_config(CATEGORY_ATTRS_PATH)
SPECIAL_RULES = load_json_config(SPECIAL_RULES_PATH)

# =================================================

# 初始化豆包客户端（兼容OpenAI格式）[citation:1]
client = openai.OpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_BASE_URL
)
print(f"doubao-client模型加载完成")

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
print(f"clip模型加载完成")

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
                "desc": value.get("desc"),
                "category": "OperationalDesignDomain",
                "has_children": len(value.get("children", [])) > 0
            }

        # 添加子节点
        for child in value.get("children", []):
            if child not in all_nodes:
                child_value = concept_data.get(child, {})
                all_nodes[child] = {
                    "name": child,
                    "desc": value.get("desc"),
                    "parent": key,  # 先临时设置父节点
                    "category": key,  # 类别就是父节点
                    "has_children": len(child_value.get("children", [])) > 0
                }

    return all_nodes

# ==================== 辅助函数：属性推断 ====================
def get_attributes_for_concept(concept: str, category: str) -> List[str]:
    """
    根据概念名称和类别返回建议的属性维度列表。
    从 JSON 配置文件中加载数据。
    """
    # 默认属性（保底使用）
    default_attrs = ["强度", "场景类型", "时间段", "影响程度", "空间分布", "时间演化"]

    # 1. 优先使用 special_rules 中的精细化属性
    if concept in SPECIAL_RULES:
        return SPECIAL_RULES[concept]

    # 2. 否则使用 category_attributes 中的类别属性
    if category in CATEGORY_ATTRIBUTES:
        return CATEGORY_ATTRIBUTES[category]

    # 3. 如果都没有，返回默认属性
    print(f"⚠️ 未找到概念 '{concept}' 或类别 '{category}' 的属性配置，使用默认属性")
    return default_attrs
# def get_attributes_for_concept(concept: str, category: str) -> List[str]:
#     """
#     根据概念名称和类别返回建议的属性维度列表。
#     """
#     # 基于类别的基本属性模板
#     category_attributes = {
#         "Weather": ["强度", "时间段", "能见度", "伴随现象"],
#         "Illumination": ["光照强度", "光源类型", "时间段", "阴影"],
#         "Precipitation": ["降水类型", "强度", "持续时间", "影响"],
#         "Fog": ["浓度", "能见度", "时间段", "高度"],
#         "RoadType": ["道路类型", "车道数", "限速", "隔离带"],
#         "RoadSurface": ["表面材质", "湿滑程度", "损坏情况", "摩擦系数"],
#         "TrafficParticipants": ["类型", "数量", "运动状态", "位置"],
#         "Vehicle": ["车辆类型", "速度", "灯光状态", "行驶意图"],
#         "VulnerableRoadUser": ["类型", "行为", "可见性", "保护装备"],
#         "DrivingDynamics": ["速度范围", "加速度", "方向变化", "紧急程度"],
#         "Infrastructure": ["类型", "状态", "可见性", "功能性"],
#         "TemporalConditions": ["具体时间", "季节", "节假日", "特殊事件"],
#         "LegalRegulatoryConditions": ["规则类型", "限制值", "适用区域", "执行力度"],
#         "VehicleCapabilities": ["能力类型", "性能参数", "可用性", "可靠性"],
#         "Perception": ["感知对象", "距离", "准确率", "环境条件"],
#         "Map": ["地图类型", "精度", "更新频率", "覆盖区域"],
#         "Traffic": ["流量状态", "密度", "组成", "规则符合性"],
#         "Road": ["道路特征", "几何", "标线", "附属设施"],
#         "Environment": ["环境因素", "综合影响"],
#         "OperationalDesignDomain": ["综合场景", "运行条件", "限制因素"],
#     }
#     # 默认属性
#     default_attrs = ["强度", "场景类型", "时间段", "影响程度"]
#
#     # 如果category在映射中，返回对应属性；否则返回默认
#     attrs = category_attributes.get(category, default_attrs)
#
#     # 可根据特定概念微调
#     special_rules = {
#         "Clear": ["云量", "能见度", "光照条件", "降水"],
#         "Cloudy": ["云量", "光照", "降水可能"],
#         "Rain": ["降雨强度", "持续时间", "路面湿滑", "能见度"],
#         "Snow": ["降雪强度", "积雪厚度", "路面结冰", "能见度"],
#         "Daylight": ["时间", "太阳高度", "光照强度", "阴影"],
#         "Night": ["月光", "人工照明", "可见度", "时段"],
#         "Dry": ["路面干燥程度", "温度", "灰尘"],
#         "Wet": ["积水程度", "潮湿", "反光", "滑移风险"],
#         "Icy": ["冰层厚度", "黑冰", "温度", "摩擦系数"],
#     }
#     if concept in special_rules:
#         attrs = special_rules[concept]
#     return attrs


# ==================== 生成描述的提示词模板 ====================
def build_prompt(concept: str, category: str, attributes: List[str]) -> str:
    """
    构建发送给豆包模型的提示词，要求生成多样化的自动驾驶场景描述。
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


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError))
)
def call_llm(prompt: str) -> List[str]:
    """
    调用豆包API生成描述，返回字符串列表。[citation:1]
    包含自动重试机制，处理限流和临时错误。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的自动驾驶场景描述生成器，严格按格式输出JSON列表。"},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1,
            stop=None
        )
        content = response.choices[0].message.content.strip()

        # 尝试解析 JSON 列表
        try:
            # 查找可能被 Markdown 代码块包裹的 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            descriptions = json.loads(content)
            if isinstance(descriptions, list):
                return descriptions
            else:
                print(f"返回格式不是列表: {content[:100]}")
                return []
        except json.JSONDecodeError:
            # 如果返回的不是合法 JSON，尝试按行分割
            lines = [line.strip("- ").strip() for line in content.split("\n") if line.strip() and not line.startswith("```")]
            # 过滤掉可能的解释性文字
            descriptions = [line for line in lines if len(line) > 10 and not line.startswith("请") and not line.startswith("以下")]
            return descriptions
    except Exception as e:
        print(f"豆包API调用失败: {e}")
        raise  # 触发重试


# ==================== CLIP 编码函数 ====================
def encode_texts(texts: List[str]) -> np.ndarray:
    """
    使用 CLIP 模型编码文本列表，返回归一化后的向量数组 (n, dim)。
    """
    if not texts:
        return None
    inputs = clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2 归一化
    return embeddings.cpu().numpy()


# ==================== 主流程 ====================
def main():
    # 1. 加载 JSON 概念树
    print(f"加载概念文件: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        concept_data = json.load(f)

    # 提取所有概念及其父类别
    concepts_dict = extract_concepts_two_pass(concept_data)
    concepts = list(concepts_dict.values())
    print(f"共加载 {len(concepts)} 个概念。")

    # 2. 为每个概念生成描述并编码
    failed_concepts = []
    successful_concepts = []
    # 存放所有embedding
    successful_embeddings = {}

    # 测试模式：只处理第一个概念
    test_mode = True  # 设为 False 时处理全部概念
    concepts_to_process = concepts[:1] if test_mode else concepts
    print(f"测试概念: {concepts_to_process[0]['name']} (类别: {concepts_to_process[0]['category']})")

    for idx, item in enumerate(tqdm(concepts_to_process, desc="处理概念")):
        concept = item["name"]
        category = item["category"]
        desc = item["desc"]

        print(f"\n[{idx+1}/{len(concepts)}] 处理概念: {concept} (类别: {category})")

        # 推断属性
        attributes = get_attributes_for_concept(concept, category)

        # 构建提示词
        prompt = build_prompt(concept, category, attributes)

        # 调用豆包API生成描述
        try:
            descriptions = call_llm(prompt)
            if not descriptions:
                print(f"警告：概念 {concept} 未生成描述，跳过。")
                failed_concepts.append(concept)
                continue

            # 取前 NUM_DESCRIPTIONS_PER_CONCEPT 条
            descriptions = descriptions[:NUM_DESCRIPTIONS_PER_CONCEPT]
            print(f"  生成 {len(descriptions)} 条描述")

            # 编码
            vecs = encode_texts(descriptions)
            if vecs is None or len(vecs) == 0:
                print(f"警告：概念 {concept} 编码失败。")
                failed_concepts.append(concept)
                continue

            # 取均值作为概念向量
            concept_vector = vecs.mean(axis=0)
            # 再次归一化
            concept_vector = concept_vector / np.linalg.norm(concept_vector)
            # concept_vectors[concept] = concept_vector
            concept_vectors = {}
            concept_vectors["name"] = concept
            name_emb = encode_texts(concept)
            concept_vectors["name_emb_key"] = f"name_emb_{idx}"
            concept_vectors["desc"] = desc
            desc_emb = encode_texts(desc)
            concept_vectors["desc_emb_key"] = f"desc_emb_{idx}"
            concept_vectors["extend_desc"] = descriptions
            concept_vectors["extend_desc_emb_key"] = [f"extend_emb_{idx}_{i}" for i in range(len(vecs))]
            concept_vectors["desc_mean_emb_key"] = f"desc_mean_emb_{idx}"
            successful_concepts.append(concept)

            successful_embeddings[f"name_emb_{idx}"] = name_emb
            successful_embeddings[f"desc_emb_{idx}"] = desc_emb
            for i, emb in enumerate(vecs):
                successful_embeddings[f"extend_emb_{idx}_{i}"] = emb
            successful_embeddings[f"desc_mean_emb_{idx}"] = concept_vector

        except Exception as e:
            print(f"概念 {concept} 处理失败: {e}")
            failed_concepts.append(concept)

        # 避免 API 限流，适当延时
        time.sleep(0.5)

    # 3. 保存结果
    if concept_vectors:
        with open(CONCEPT_EXTEND_OUTPUT_NAMES, "w", encoding="utf-8") as f:
            json.dump(successful_concepts, f, ensure_ascii=False, indent=2)
        print(f"✅ 概念向量已保存至: {CONCEPT_EXTEND_OUTPUT_NAMES}")
        print(f"  包含 {len(successful_concepts)} 个概念")

        # 保存嵌入向量到npz（压缩格式）
        np.savez_compressed(CONCEPT_EXTEND_EMBEDDING_OUTPUT_NAMES, **successful_embeddings)
        # concept_names = list(concept_vectors.keys())
        # vectors = np.array([concept_vectors[name] for name in concept_names])
        #
        # np.save(OUTPUT_VECTORS, vectors)
        # with open(OUTPUT_NAMES, "w", encoding="utf-8") as f:
        #     json.dump(concept_names, f, indent=2)
        #
        # print(f"\n✅ 成功处理 {len(concept_names)} 个概念，失败 {len(failed_concepts)} 个。")
        # if failed_concepts:
        #     print("❌ 失败概念:", failed_concepts)
        # print(f"📊 向量维度: {vectors.shape[1]}")
        # print(f"💾 向量库已保存至 {OUTPUT_VECTORS} 和 {OUTPUT_NAMES}")
    else:
        print("❌ 没有成功处理任何概念，请检查 API 配置和网络连接。")

if __name__ == "__main__":
    main()