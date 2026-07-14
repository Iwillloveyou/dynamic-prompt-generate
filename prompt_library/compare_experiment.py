#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三组对比实验脚本 (论文 3.4.2)
- 实验1: 仅 OpenODD 关键词 (无 LLM 扩展)
- 实验2: WordNet 通用词扩展
- 实验3: LLM 自由生成 (无本体结构限定)

计算除人工一致评分和幻觉率以外的所有指标。
"""

import json
import os
import pickle
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer

# NLP 工具
import nltk
from nltk.corpus import wordnet as wn

# 嵌入模型 (使用 Sentence-BERT 统一编码所有文本)
from sentence_transformers import SentenceTransformer

# 可选: OpenAI API (实验三)
import openai

# 设置随机种子保证可复现
random.seed(42)
np.random.seed(42)

# ====================== 全局配置 ======================
OPENAI_API_KEY = "" # 替换为你API Key
OPENAI_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 以官方文档为准
MODEL_NAME = "deepseek-v3-2-251201"  # 深度思考模型
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# 模型
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 轻量句向量模型
CACHE_DIR = "./exp_cache"                   # 缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)

# 实验参数
EXP2_MAX_TERMS = 200        # WordNet 扩展最多保留词数
EXP3_NUM_TERMS = 100        # LLM 自由生成词数
EXPECTED_SCENE_PER_CONCEPT = 8   # 每个概念期望的扩展描述条数

# ====================== 通用工具函数 ======================
def load_standard_ontology(json_path: str) -> Tuple[set, Dict[str, str], Dict[str, str]]:
    """加载标准 OpenODD 本体，返回概念集合、父节点字典、原始描述字典"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    concepts = set()
    parent_of = {}
    desc_of = {}
    for node in data.values():
        if isinstance(node, dict) and 'parent' in node:
            name = node.get('name')
            if name is None:
                continue
            concepts.add(name)
            p = node['parent']
            if p is not None:
                parent_of[name] = p
            desc = node.get('desc', '')
            if desc:
                desc_of[name] = desc
    # 确保根节点存在
    if 'OperationalDesignDomain' not in concepts:
        concepts.add('OperationalDesignDomain')
    return concepts, parent_of, desc_of

def embed_texts(texts: List[str], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """批量嵌入文本，返回 {text: vector} 字典"""
    if not texts:
        return {}
    vecs = model.encode(texts, show_progress_bar=False)
    return {text: vec for text, vec in zip(texts, vecs)}

def cosine_sim(a, b):
    return 1 - cosine(a, b)

def build_tree_structure(items: List[Dict]) -> Tuple[Dict[str, str], bool, Dict[str, int], int, float, int, float]:
    """
    从 items 中提取 parent 字段构建树
    返回: parent_of, acyclic, depth_dict, max_depth, avg_depth, max_fanout, avg_fanout
    """
    parent_of = {}
    for item in items:
        name = item.get('name')
        parent = item.get('parent')
        if name and parent:
            parent_of[name] = parent
    # 无环检测
    def is_acyclic(parent_of):
        graph = defaultdict(list)
        for child, p in parent_of.items():
            graph[p].append(child)
        state = {}
        def dfs(node):
            state[node] = 1
            for nei in graph[node]:
                if state.get(nei, 0) == 1:
                    return True
                if state.get(nei, 0) == 0 and dfs(nei):
                    return True
            state[node] = 2
            return False
        for node in list(graph.keys()):
            if state.get(node, 0) == 0:
                if dfs(node):
                    return False
        return True
    acyclic = is_acyclic(parent_of)
    # 深度和扇出
    children_of = defaultdict(list)
    for child, p in parent_of.items():
        children_of[p].append(child)
    # 找根
    all_parents = set(parent_of.values())
    all_children = set(parent_of.keys())
    roots = all_parents - all_children
    root = next(iter(roots)) if roots else None
    depth = {}
    if root is not None:
        depth[root] = 0
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in children_of[node]:
                depth[child] = depth[node] + 1
                queue.append(child)
    depths = list(depth.values())
    max_depth = max(depths) if depths else 0
    avg_depth = np.mean(depths) if depths else 0
    # 扇出: 每个内部节点的子节点数
    fanouts = [len(c) for c in children_of.values() if c]
    max_fanout = max(fanouts) if fanouts else 0
    avg_fanout = np.mean(fanouts) if fanouts else 0
    return parent_of, acyclic, depth, max_depth, avg_depth, max_fanout, avg_fanout

def compute_edge_preservation(ext_parent_of: Dict[str, str], std_parent_of: Dict[str, str]) -> float:
    consistent = 0
    total = len(std_parent_of)
    if total == 0:
        return 0.0
    for child, std_p in std_parent_of.items():
        if child in ext_parent_of and ext_parent_of[child] == std_p:
            consistent += 1
    return consistent / total

def compute_coverage(ext_concepts: set, std_concepts: set) -> float:
    if not std_concepts:
        return 0.0
    covered = len(ext_concepts.intersection(std_concepts))
    return covered / len(std_concepts)

def compute_clip_similarities(ext_data: List[Dict], emb_dict: Dict[str, np.ndarray]):
    """
    计算概念一致性 S_con 和簇内相似度 S_pair
    注意: 这里使用传入的 emb_dict (任意嵌入模型, 论文中为 CLIP)
    """
    s_con_list = []
    s_pair_list = []
    for item in ext_data:
        desc_key = item.get('desc_emb_key')
        ext_keys = item.get('extend_desc_emb_key', [])
        if not desc_key or desc_key not in emb_dict:
            continue
        if not ext_keys:
            continue
        desc_vec = emb_dict[desc_key]
        ext_vecs = [emb_dict[k] for k in ext_keys if k in emb_dict]
        if not ext_vecs:
            continue
        # 概念一致性
        sims = [cosine_sim(desc_vec, ev) for ev in ext_vecs]
        s_con_list.append(np.mean(sims))
        # 簇内相似度
        if len(ext_vecs) >= 2:
            pair_sims = []
            for i in range(len(ext_vecs)):
                for j in range(i+1, len(ext_vecs)):
                    pair_sims.append(cosine_sim(ext_vecs[i], ext_vecs[j]))
            s_pair_list.append(np.mean(pair_sims))
        else:
            s_pair_list.append(1.0)
    s_con = np.mean(s_con_list) if s_con_list else 0.0
    s_pair = np.mean(s_pair_list) if s_pair_list else 0.0
    return s_con, s_pair

def compute_lexical_diversity(ext_data: List[Dict]) -> Tuple[float, float]:
    """描述多样性得分 D_lex 和平均词汇熵 H_word"""
    all_sentences = []
    concept_texts = []
    for item in ext_data:
        ext_list = item.get('extend_desc', [])
        if not ext_list:
            continue
        all_sentences.extend(ext_list)
        combined = ' '.join(ext_list)
        concept_texts.append(combined)
    # D_lex: 唯一 bigram 比例
    if all_sentences:
        vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word', lowercase=True)
        try:
            X = vectorizer.fit_transform(all_sentences)
            total_bigrams = X.sum()
            unique_bigrams = len(vectorizer.get_feature_names_out())
            D_lex = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0
        except:
            D_lex = 0.0
    else:
        D_lex = 0.0
    # H_word: 每个概念的 unigram 熵平均
    entropy_list = []
    for text in concept_texts:
        tokens = text.lower().split()
        if not tokens:
            continue
        freq = Counter(tokens)
        total = len(tokens)
        probs = [cnt/total for cnt in freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs)
        entropy_list.append(entropy)
    H_word = np.mean(entropy_list) if entropy_list else 0.0
    return D_lex, H_word

def compute_bert_similarity(ext_data: List[Dict], std_desc_of: Dict[str, str], model: SentenceTransformer) -> float:
    """使用 BERT 嵌入计算扩展描述与标准描述的语义相似度 (Sim_BERT)"""
    sim_sum = 0.0
    count = 0
    for item in ext_data:
        concept = item['name']
        orig_desc = std_desc_of.get(concept, '')
        if not orig_desc:
            continue
        orig_vec = model.encode(orig_desc, show_progress_bar=False)
        ext_desc_list = item.get('extend_desc', [])
        for ext in ext_desc_list:
            ext_vec = model.encode(ext, show_progress_bar=False)
            sim = cosine_sim(orig_vec, ext_vec)
            sim_sum += sim
            count += 1
    return sim_sum / count if count > 0 else 0.0

def compute_relation_integrity(ext_data: List[Dict], std_parent_of: Dict[str, str],
                               std_concepts: set, expected_scene_per_concept: int,
                               emb_dict: Dict[str, np.ndarray]) -> float:
    """
    关系完整度 η_rel: 综合父子关系、概念定义、场景描述数量、嵌入键完整性
    """
    # 1. 父子关系正确数
    ext_parent_of, _, _, _, _, _, _ = build_tree_structure(ext_data)
    correct_parents = 0
    total_parents = len(std_parent_of)
    for child, std_p in std_parent_of.items():
        if child in ext_parent_of and ext_parent_of[child] == std_p:
            correct_parents += 1
    # 2. 概念定义关系 (desc 完全匹配原始描述, 且 desc_emb_key 有效)
    correct_def = 0
    total_def = len(std_concepts)
    for item in ext_data:
        name = item['name']
        std_desc = std_desc_of.get(name, '')
        if item.get('desc', '') == std_desc and item.get('desc_emb_key') in emb_dict:
            correct_def += 1
    # 3. 场景描述关系 (数量达到目标且每条非空)
    correct_scene_count = 0
    total_scene_expected = len(std_concepts) * expected_scene_per_concept
    for item in ext_data:
        ext_list = item.get('extend_desc', [])
        if len(ext_list) == expected_scene_per_concept and all(isinstance(s, str) and s.strip() for s in ext_list):
            correct_scene_count += expected_scene_per_concept
        else:
            # 未达标部分不计入正确
            pass
    # 4. 嵌入绑定关系
    correct_embed = 0
    total_embed_expected = len(std_concepts) * (1 + expected_scene_per_concept)
    for item in ext_data:
        # desc embedding
        if item.get('desc_emb_key') in emb_dict:
            correct_embed += 1
        # scene embeddings
        ext_keys = item.get('extend_desc_emb_key', [])
        ext_desc = item.get('extend_desc', [])
        if len(ext_keys) == len(ext_desc) and all(k in emb_dict for k in ext_keys):
            correct_embed += len(ext_keys)
    total_correct = correct_parents + correct_def + correct_scene_count + correct_embed
    total_expected = total_parents + total_def + total_scene_expected + total_embed_expected
    return total_correct / total_expected if total_expected > 0 else 0.0

def save_extended_data(ext_data: List[Dict], json_path: str, emb_dict: Dict[str, np.ndarray], npz_path: str):
    """保存扩展JSON和嵌入npz文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ext_data, f, indent=2, ensure_ascii=False)
    np.savez(npz_path, **emb_dict)
    print(f"Saved: {json_path}, {npz_path}")

# ====================== 实验一: 仅 OpenODD 关键词 ======================
def build_experiment1(std_json_path: str, model: SentenceTransformer) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """构建实验一提示库: 无扩展描述, 仅用概念名作为唯一描述"""
    std_concepts, std_parent_of, std_desc_of = load_standard_ontology(std_json_path)
    ext_data = []
    emb_dict = {}
    for concept in std_concepts:
        parent = std_parent_of.get(concept, None)
        desc = std_desc_of.get(concept, "")
        # 没有 extend_desc
        item = {
            "name": concept,
            "parent": parent,
            "desc": desc,
            "desc_emb_key": f"{concept}_desc_emb",
            "extend_desc": [],          # 无扩展
            "extend_desc_emb_key": [],
            "desc_mean_emb_key": None   # 不需要
        }
        ext_data.append(item)
        # 生成嵌入
        desc_vec = model.encode(desc, show_progress_bar=False) if desc else np.zeros(model.get_sentence_embedding_dimension())
        emb_dict[item["desc_emb_key"]] = desc_vec
    return ext_data, emb_dict

# ====================== 实验二: WordNet 通用词扩展 ======================
def get_wordnet_related_terms(seed_concepts: List[str], max_terms: int = 200) -> List[str]:
    """
    从种子概念获取 WordNet 同义词、上位词、下位词，过滤去重后返回最多 max_terms 个词
    """
    terms = set()
    for concept in seed_concepts:
        # 词形归一化
        synsets = wn.synsets(concept.lower())
        if not synsets:
            continue
        for syn in synsets:
            # 同义词
            for lemma in syn.lemmas():
                term = lemma.name().replace('_', ' ')
                terms.add(term.lower())
            # 上位词
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    term = lemma.name().replace('_', ' ')
                    terms.add(term.lower())
            # 下位词
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    term = lemma.name().replace('_', ' ')
                    terms.add(term.lower())
    # 过滤长尾或无关词 (可简单按长度)
    terms = {t for t in terms if 2 <= len(t.split()) <= 6}
    # 限制数量
    terms = list(terms)
    if len(terms) > max_terms:
        random.shuffle(terms)
        terms = terms[:max_terms]
    return terms

def generate_descriptions_for_terms(terms: List[str], model: SentenceTransformer,
                                    use_llm: bool = False, llm_template: str = None) -> Dict[str, List[str]]:
    """
    为每个词生成扩展描述 (8条)
    如果 use_llm=True，调用 GPT-4 生成；否则使用简单模板生成占位描述
    """
    # 简单模板 (用于快速测试，不依赖 API)
    default_template = lambda term: [
        f"A typical driving scenario involving {term} on a clear day.",
        f"Moderate traffic conditions with {term} present on urban roads.",
        f"Nighttime operation with {term} under street lighting.",
        f"Adverse weather conditions such as rain and {term}.",
        f"Highway driving scenario featuring {term}.",
        f"Intersection crossing with {term} as a relevant factor.",
        f"Parking maneuver in an environment with {term}.",
        f"Rural road with {term} affecting vehicle dynamics."
    ]
    result = {}
    for term in terms:
        if use_llm:
            # 调用 LLM 生成 8 条场景描述
            prompt = f"You are a senior autonomous driving scenario engineer responsible for generating high-quality image-text paired training data for autonomous driving datasets. Generate 8 diverse short scenario descriptions (each 10-20 words) for the autonomous driving operational design domain concept '{term}'. Output as a numbered list."
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                text = response.choices[0].message.content
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                descs = []
                for line in lines:
                    if line and line[0].isdigit() and '.' in line:
                        descs.append(line.split('.', 1)[-1].strip())
                # 补足到8条
                if len(descs) < 8:
                    descs += default_template(term)[:8 - len(descs)]
                descs = descs[:8]
                # 存入字典，不要提前return
                result[term] = descs
            except Exception as e:
                print(f"LLM error for {term}: {e}, using template")
                result[term] = default_template(term)
        else:
            result[term] = default_template(term)
        print(f"为{term} 生成 {len(result[term])} 条描述")
    # 全部遍历完成后再返回字典
    print(f"总词数：{len(terms)}, 成功生成描述词数：{len(result)}")
    return result

def build_experiment2(std_json_path: str, model: SentenceTransformer,
                      max_terms=200, use_llm=False) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    构建 WordNet 扩展提示库
    """
    std_concepts, std_parent_of, std_desc_of = load_standard_ontology(std_json_path)
    # 获取种子概念 (所有标准概念)
    seed = list(std_concepts)
    # 通过 WordNet 扩展
    expanded_terms = get_wordnet_related_terms(seed, max_terms=max_terms)
    print(f"Experiment2: obtained {len(expanded_terms)} terms from WordNet")
    # 生成扩展描述
    term_to_descs = generate_descriptions_for_terms(expanded_terms, model, use_llm=use_llm)
    # 构建 ext_data (简单树结构: 根节点为 "WordNetRoot"，每个术语为直接子节点)
    ext_data = []
    emb_dict = {}
    root_item = {
        "name": "WordNetRoot",
        "parent": None,
        "desc": "Root node for WordNet expanded concepts",
        "desc_emb_key": "WordNetRoot_desc_emb",
        "extend_desc": [],
        "extend_desc_emb_key": [],
        "desc_mean_emb_key": None
    }
    ext_data.append(root_item)
    root_desc_vec = model.encode(root_item["desc"], show_progress_bar=False)
    emb_dict[root_item["desc_emb_key"]] = root_desc_vec

    for term in expanded_terms:
        descs = term_to_descs.get(term, [])
        desc_text = f"Concept related to {term} in driving context."
        item = {
            "name": term,
            "parent": "WordNetRoot",
            "desc": desc_text,
            "desc_emb_key": f"{term}_desc_emb",
            "extend_desc": descs,
            "extend_desc_emb_key": [f"{term}_ext_{i}" for i in range(len(descs))],
            "desc_mean_emb_key": None
        }
        ext_data.append(item)
        # 嵌入
        desc_vec = model.encode(desc_text, show_progress_bar=False)
        emb_dict[item["desc_emb_key"]] = desc_vec
        for i, d in enumerate(descs):
            ext_vec = model.encode(d, show_progress_bar=False)
            emb_dict[f"{term}_ext_{i}"] = ext_vec
    return ext_data, emb_dict

# ====================== 实验三: LLM 自由生成 ======================
def llm_generate_terms_and_descriptions(num_terms=100, use_real_llm=False) -> Dict[str, List[str]]:
    """
    使用 LLM 生成 num_terms 个自动驾驶相关术语，并为每个生成 8 条描述
    如果 use_real_llm=False，返回模拟数据
    """
    if use_real_llm and openai.api_key:
        # 第一步: 生成术语列表
        prompt_terms = f"Generate {num_terms} diverse terms related to autonomous driving Operational Design Domain (ODD). Output as a comma-separated list without numbering."
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_terms}],
                temperature=0.8,
                max_tokens=500
            )
            text = response.choices[0].message.content
            terms = [t.strip() for t in text.split(',')]
            terms = terms[:num_terms]
        except Exception as e:
            print(f"LLM term generation failed: {e}, using mock terms")
            terms = [f"AutoTerm_{i}" for i in range(num_terms)]
    else:
        # 模拟术语
        terms = [f"ODDConcept_{i}" for i in range(num_terms)]
    # 为每个术语生成描述
    result = generate_descriptions_for_terms(terms, use_llm=use_real_llm)
    return result

def build_experiment3(model: SentenceTransformer, num_terms=100, use_llm=False) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """构建实验三提示库: 无本体结构，扁平列表"""
    term_to_descs = llm_generate_terms_and_descriptions(num_terms, use_real_llm=use_llm)
    ext_data = []
    emb_dict = {}
    for term, descs in term_to_descs.items():
        desc_text = f"Term related to autonomous driving ODD: {term}"
        item = {
            "name": term,
            "parent": None,        # 无结构
            "desc": desc_text,
            "desc_emb_key": f"{term}_desc_emb",
            "extend_desc": descs,
            "extend_desc_emb_key": [f"{term}_ext_{i}" for i in range(len(descs))],
            "desc_mean_emb_key": None
        }
        ext_data.append(item)
        # 嵌入
        desc_vec = model.encode(desc_text, show_progress_bar=False)
        emb_dict[item["desc_emb_key"]] = desc_vec
        for i, d in enumerate(descs):
            ext_vec = model.encode(d, show_progress_bar=False)
            emb_dict[f"{term}_ext_{i}"] = ext_vec
    return ext_data, emb_dict

# ====================== 主评估函数 ======================
def evaluate_experiment(ext_data: List[Dict], emb_dict: Dict[str, np.ndarray],
                        std_concepts: set, std_parent_of: Dict[str, str],
                        std_desc_of: Dict[str, str], model: SentenceTransformer,
                        exp_name: str) -> Dict[str, Any]:
    """对某一实验的 ext_data 和 emb_dict 计算各项指标"""
    print(f"\n========== Evaluating {exp_name} ==========")
    # 1. 结构完整性
    ext_parent_of, acyclic, depth, max_depth, avg_depth, max_fanout, avg_fanout = build_tree_structure(ext_data)
    edge_pres = compute_edge_preservation(ext_parent_of, std_parent_of) if std_parent_of else 0.0
    print(f"Edge preservation: {edge_pres:.4f}, Acyclic: {acyclic}")
    # 2. 层级深度、分支广度
    print(f"Max depth: {max_depth}, Avg depth: {avg_depth:.4f}, Max fanout: {max_fanout}, Avg fanout: {avg_fanout:.4f}")
    # 3. 知识覆盖
    ext_concepts = {item['name'] for item in ext_data}
    cov = compute_coverage(ext_concepts, std_concepts)
    print(f"Concept coverage: {cov:.4f}")
    # 4. 语义一致 (需要使用 embedding)
    s_con, s_pair = compute_clip_similarities(ext_data, emb_dict)
    print(f"S_con: {s_con:.4f}, S_pair: {s_pair:.4f}")
    # 5. 语义多样
    d_lex, h_word = compute_lexical_diversity(ext_data)
    print(f"D_lex: {d_lex:.4f}, H_word: {h_word:.4f}")
    # 6. 语义相似度 (BERT)
    sim_bert = compute_bert_similarity(ext_data, std_desc_of, model)
    print(f"Sim_BERT: {sim_bert:.4f}")
    # 7. 关系完整度
    eta_rel = compute_relation_integrity(ext_data, std_parent_of, std_concepts,
                                         EXPECTED_SCENE_PER_CONCEPT, emb_dict)
    print(f"Relation integrity η_rel: {eta_rel:.4f}")
    # 收集结果
    results = {
        "edge_preservation": edge_pres,
        "acyclic": acyclic,
        "max_depth": max_depth,
        "avg_depth": avg_depth,
        "max_fanout": max_fanout,
        "avg_fanout": avg_fanout,
        "concept_coverage": cov,
        "S_con": s_con,
        "S_pair": s_pair,
        "D_lex": d_lex,
        "H_word": h_word,
        "Sim_BERT": sim_bert,
        "eta_rel": eta_rel
    }
    return results

# ====================== 主程序 ======================
def main():
    # 配置路径
    STANDARD_JSON = "openodd_desc.json"   # 请修改为实际路径
    # 加载标准本体
    std_concepts, std_parent_of, std_desc_of = load_standard_ontology(STANDARD_JSON)
    # 嵌入模型
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # 实验一
    exp1_data, exp1_emb = build_experiment1(STANDARD_JSON, model)
    res1 = evaluate_experiment(exp1_data, exp1_emb, std_concepts, std_parent_of, std_desc_of, model, "Experiment1 (Only Keywords)")
    # 实验二 (WordNet, 使用模板生成描述，不调用LLM以节省时间。如需LLM生成，设置use_llm=True)
    exp2_data, exp2_emb = build_experiment2(STANDARD_JSON, model, max_terms=EXP2_MAX_TERMS, use_llm=True)
    res2 = evaluate_experiment(exp2_data, exp2_emb, std_concepts, std_parent_of, std_desc_of, model, "Experiment2 (WordNet Expansion)")
    # 实验三 (LLM自由生成, 模拟模式; 如需真实LLM，设置use_llm=True并确保API key)
    exp3_data, exp3_emb = build_experiment3(model, num_terms=EXP3_NUM_TERMS, use_llm=True)
    res3 = evaluate_experiment(exp3_data, exp3_emb, std_concepts, std_parent_of, std_desc_of, model, "Experiment3 (LLM Free Generation)")

    # 汇总表格
    print("\n========== Summary of Three Experiments ==========")
    print("{:<25} | {:>8} | {:>8} | {:>8}".format("Metric", "Exp1", "Exp2", "Exp3"))
    print("-" * 60)
    for metric in ["edge_preservation", "concept_coverage", "eta_rel", "Sim_BERT", "S_con", "S_pair", "D_lex", "H_word", "max_depth", "max_fanout"]:
        v1 = res1.get(metric, 0.0)
        v2 = res2.get(metric, 0.0)
        v3 = res3.get(metric, 0.0)
        if isinstance(v1, float):
            print("{:<25} | {:>8.4f} | {:>8.4f} | {:>8.4f}".format(metric, v1, v2, v3))
        else:
            print("{:<25} | {:>8} | {:>8} | {:>8}".format(metric, v1, v2, v3))

if __name__ == "__main__":
    main()
    # expanded_terms =["Environment", "Road"]
    # model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # term_to_descs = generate_descriptions_for_terms(expanded_terms, model, use_llm=True)
    # for key, desc_list in term_to_descs.items():
    #     print(f"【{key}】")
    # for idx, item in enumerate(desc_list):
    #     print(f"  {idx+1}. {item}")
    # print()