import os
import json
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from typing import Dict, List, Tuple, Any

# 可选：使用 sentence-transformers 计算 BERT 相似度
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
try:
    from sentence_transformers import SentenceTransformer
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. BERT similarity will be approximated with CLIP embeddings.")

# ====================== 1. 加载数据 ======================
def load_standard_ontology(json_path: str) -> Dict[str, Any]:
    """加载标准OpenODD本体，返回概念集合、父子关系、原始描述"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有节点（每个json对象是一个概念）
    concepts = set()
    parent_of = {}       # child -> parent
    desc_of = {}         # concept -> desc
    # ✅ 正确遍历：键是概念名，值是内容
    for concept_name, node_info in data.items():
        # 加入概念集合
        concepts.add(concept_name)

        # 父节点
        parent = node_info.get("parent", None)
        if parent is not None:
            parent_of[concept_name] = parent

        # 描述
        desc = node_info.get("desc", "")
        desc_of[concept_name] = desc
    # 确保根节点也在集合中
    if 'OperationalDesignDomain' not in concepts:
        concepts.add('OperationalDesignDomain')
    return concepts, parent_of, desc_of

def load_extended_data(json_path: str, npz_path: str) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """加载扩展json和嵌入npz"""
    with open(json_path, 'r', encoding='utf-8') as f:
        ext_data = json.load(f)   # 列表，每个元素是一个概念节点
    embeddings = np.load(npz_path, allow_pickle=True)
    # 将npz转换为普通字典 {key: vector}
    emb_dict = {key: embeddings[key] for key in embeddings.files}
    return ext_data, emb_dict

# ====================== 2. 辅助函数 ======================
def cosine_sim(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return 1 - cosine(a, b)

def compute_depth_and_fanout(parent_of: Dict[str, str]) -> Tuple[Dict[str, int], int, float, int, float]:
    """
    根据父子关系计算深度和扇出
    返回：深度字典、最大深度、平均深度、最大扇出、平均扇出
    """
    # 构建孩子列表
    children_of = defaultdict(list)
    root = None
    for child, parent in parent_of.items():
        children_of[parent].append(child)
        if parent not in parent_of:
            root = parent   # 根节点（没有父节点）
    if root is None and len(parent_of) > 0:
        # 找到没有父节点的节点
        all_parents = set(parent_of.values())
        all_children = set(parent_of.keys())
        roots = all_parents - all_children
        if roots:
            root = next(iter(roots))
    # 计算深度（BFS）
    depth = {}
    if root is not None:
        depth[root] = 0
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in children_of[node]:
                depth[child] = depth[node] + 1
                queue.append(child)
    # 平均深度
    depths = list(depth.values())
    max_depth = max(depths) if depths else 0
    avg_depth = np.mean(depths) if depths else 0
    # 扇出：每个内部节点的子节点数
    fanouts = [len(children) for children in children_of.values() if children]
    max_fanout = max(fanouts) if fanouts else 0
    avg_fanout = np.mean(fanouts) if fanouts else 0
    return depth, max_depth, avg_depth, max_fanout, avg_fanout

def check_acyclic(parent_of: Dict[str, str]) -> bool:
    """通过DFS检查是否有环（有向图）"""
    graph = defaultdict(list)
    for child, parent in parent_of.items():
        graph[parent].append(child)
    # 检测环：访问状态 0=未访问,1=访问中,2=已处理
    state = defaultdict(int)
    def dfs(node):
        state[node] = 1
        for nei in graph[node]:
            if state[nei] == 1:
                return True   # 有环
            if state[nei] == 0 and dfs(nei):
                return True
        state[node] = 2
        return False
    for node in list(graph.keys()):
        if state[node] == 0 and dfs(node):
            return False
    return True

def compute_relation_integrity(ext_data, emb_dict, std_parent_of, ext_parent_of, std_desc_of, std_concepts, expected_scene_per_concept=8):
    # 1. 父子关系
    correct_parents = sum(1 for child, p in ext_parent_of.items()
                          if child in std_parent_of and std_parent_of[child] == p)
    total_parents = len(std_parent_of)

    # 2. 概念定义关系（desc完全匹配）：扩展库中 desc 与标准相同且 desc_emb_key 有效数目占预期的比例，预期=标准本体中概念个数（例如 110）
    correct_def = 0
    for item in ext_data:
        std_desc = std_desc_of.get(item['name'], '')
        if item.get('desc', '') == std_desc and item.get('desc_emb_key') in emb_dict:
            correct_def += 1
    total_def = len(std_concepts)

    # 3. 场景描述关系（数量达标）: 扩展库中 extend_desc 数组长度达到目标值且每条非空数目占预期的比例，预期=概念个数 × 每条期望的场景描述数（例如 110 × 8 = 880）
    correct_scene_count = 0
    total_scene_expected = len(std_concepts) * expected_scene_per_concept
    for item in ext_data:
        ext_list = item.get('extend_desc', [])
        # 假设预期8条，这里只记录达到8条且每条非空的计数
        if len(ext_list) == expected_scene_per_concept and all(isinstance(s, str) and s.strip() for s in ext_list):
            correct_scene_count += expected_scene_per_concept
        else:
            correct_scene_count += len(ext_list)   # 折衷：按实际条数计数，但完整度会降低

    # 4. 嵌入绑定关系: 原始描述+扩展描述中有有效的嵌入键，且 desc_emb_key 有效的数目占预期的比例，预期=(1 + 每条期望场景描述数) × 概念个数 = 110 × (1+8) = 990
    correct_embed = 0
    total_embed_expected = len(std_concepts) * (1 + expected_scene_per_concept)
    for item in ext_data:
        # desc embedding
        if item.get('desc_emb_key') in emb_dict:
            correct_embed += 1
        # scene embeddings
        ext_keys = item.get('extend_desc_emb_key', [])
        if len(ext_keys) == len(item.get('extend_desc', [])) and all(k in emb_dict for k in ext_keys):
            correct_embed += len(ext_keys)

    total_correct = correct_parents + correct_def + correct_scene_count + correct_embed
    total_expected = total_parents + total_def + total_scene_expected + total_embed_expected
    return total_correct / total_expected if total_expected > 0 else 0

def compute_clip_similarities(ext_data: List[Dict], emb_dict: Dict[str, np.ndarray]):
    """
    计算概念一致性 (S_con) 和 簇内相似度 (S_pair)
    概念一致性：每个概念的每个扩展描述与原始描述的平均余弦相似度
    簇内相似度：每个概念内扩展描述两两之间的平均余弦相似度
    """
    s_con_list = []      # 每个概念的平均值
    s_pair_list = []     # 每个概念的簇内平均相似度
    for item in ext_data:
        desc_emb_key = item.get('desc_emb_key')
        extend_keys = item.get('extend_desc_emb_key', [])
        if not desc_emb_key or desc_emb_key not in emb_dict:
            continue
        if not extend_keys:
            continue
        desc_vec = emb_dict[desc_emb_key]
        ext_vecs = [emb_dict[key] for key in extend_keys if key in emb_dict]
        if not ext_vecs:
            continue
        # 概念一致性：每个ext与desc的相似度平均
        sims_to_desc = [cosine_sim(desc_vec, ev) for ev in ext_vecs]
        s_con_list.append(np.mean(sims_to_desc))
        # 簇内相似度：所有对（i<j）的平均
        n = len(ext_vecs)
        if n >= 2:
            pair_sims = []
            for i in range(n):
                for j in range(i+1, n):
                    pair_sims.append(cosine_sim(ext_vecs[i], ext_vecs[j]))
            s_pair_list.append(np.mean(pair_sims))
        else:
            s_pair_list.append(1.0)  # 只有一个扩展描述时，视为完全相似
    s_con = np.mean(s_con_list) if s_con_list else 0
    s_pair = np.mean(s_pair_list) if s_pair_list else 0
    return s_con, s_pair

def compute_lexical_diversity(ext_data: List[Dict]) -> Tuple[float, float]:
    """
    计算描述多样性得分 (D_lex) 和 平均词汇熵 (H_word)
    D_lex: 所有扩展描述合并后，唯一bigram比例
    H_word: 每个概念内扩展描述合并后的unigram熵，再取平均
    """
    all_text = []          # 所有扩展描述句子
    concept_texts = []     # 每个概念的合并文本
    for item in ext_data:
        ext_desc_list = item.get('extend_desc', [])
        if not ext_desc_list:
            continue
        # 合并当前概念的所有扩展描述为一个字符串
        combined = ' '.join(ext_desc_list)
        concept_texts.append(combined)
        all_text.extend(ext_desc_list)

    # D_lex: 唯一bigram比例
    if all_text:
        vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word', lowercase=True)
        try:
            X = vectorizer.fit_transform(all_text)
            total_bigrams = X.sum()
            unique_bigrams = len(vectorizer.get_feature_names_out())
            D_lex = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
        except:
            D_lex = 0.0
    else:
        D_lex = 0.0

    # H_word: 每个概念的unigram熵平均
    entropy_list = []
    for text in concept_texts:
        tokens = text.lower().split()
        if not tokens:
            continue
        freq = Counter(tokens)
        total = len(tokens)
        probs = [count/total for count in freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs)
        entropy_list.append(entropy)
    H_word = np.mean(entropy_list) if entropy_list else 0.0
    return D_lex, H_word

def compute_bert_similarity(ext_data: List[Dict], standard_desc_of: Dict[str, str]):
    """
    计算语义相似度 (Sim_BERT)：每个概念的原始描述与每个扩展描述的BERT嵌入余弦相似度，取平均
    若BERT不可用，则使用CLIP嵌入（近似）
    """
    if BERT_AVAILABLE:
        # 收集所有需要编码的文本
        all_texts = []
        # 保持顺序：原始描述在前，然后扩展描述
        mapping = []  # (concept, is_desc, text)
        for item in ext_data:
            concept = item['name']
            orig_desc = standard_desc_of.get(concept, '')
            if orig_desc:
                mapping.append((concept, 'desc', orig_desc))
                all_texts.append(orig_desc)
            ext_desc_list = item.get('extend_desc', [])
            for ext in ext_desc_list:
                mapping.append((concept, 'ext', ext))
                all_texts.append(ext)
        # 批量编码
        embeddings = bert_model.encode(all_texts, show_progress_bar=False)
        # 计算相似度
        sim_sum = 0
        count = 0
        # 找到每个概念对应的原始描述索引和扩展描述索引
        concept_desc_idx = {}
        concept_ext_indices = defaultdict(list)
        for idx, (concept, typ, _) in enumerate(mapping):
            if typ == 'desc':
                concept_desc_idx[concept] = idx
            else:
                concept_ext_indices[concept].append(idx)
        for concept, ext_indices in concept_ext_indices.items():
            if concept not in concept_desc_idx:
                continue
            desc_vec = embeddings[concept_desc_idx[concept]]
            for ext_idx in ext_indices:
                ext_vec = embeddings[ext_idx]
                sim = cosine_sim(desc_vec, ext_vec)
                sim_sum += sim
                count += 1
        return sim_sum / count if count > 0 else 0.0
    else:
        # 回退：使用CLIP嵌入计算相似度（需要从emb_dict获取）
        warnings.warn("BERT not available, using CLIP embeddings for Sim_BERT (may differ from paper).")
        # 这里需要emb_dict中的desc_emb_key和extend_desc_emb_key
        # 由于该函数没有emb_dict参数，我们返回None并提示
        return None

# ====================== 3. 主计算流程 ======================
def main():
    # 配置文件路径（请根据实际位置修改）
    STANDARD_JSON = "openodd_desc.json"
    prompt_library_root = '../prompt_library/result/'
    EXT_JSON = os.path.join(prompt_library_root, 'concept_extend.json')
    EXT_NPZ = os.path.join(prompt_library_root, 'concept_extend.embeddings.npz')

    # 加载标准本体
    std_concepts, std_parent_of, std_desc_of = load_standard_ontology(STANDARD_JSON)
    total_std_edges = len(std_parent_of)  # 标准本体中父子边的总数（即非根节点数）
    print(f"标准本体: 概念数={len(std_concepts)}, 父子边数={total_std_edges}")

    # 加载扩展数据
    ext_data, emb_dict = load_extended_data(EXT_JSON, EXT_NPZ)

    # 构建扩展数据中的父子关系
    ext_parent_of = {}
    ext_concepts = set()
    for item in ext_data:
        name = item.get('name')
        parent = item.get('parent')
        if name:
            ext_concepts.add(name)
        if parent and name:
            ext_parent_of[name] = parent
    print(f"扩展数据: 概念数={len(ext_concepts)}")

    # ---------- 结构完整性 ----------
    # 父子边保留率：一致的父子对数 / 标准总边数
    consistent_edges = 0
    for child, std_parent in std_parent_of.items():
        if child in ext_parent_of and ext_parent_of[child] == std_parent:
            consistent_edges += 1
    edge_preservation = consistent_edges / total_std_edges if total_std_edges > 0 else 0
    print(f"父子边保留率 ρ_edge: {edge_preservation:.4f}")

    # 无环性
    acyclic = check_acyclic(ext_parent_of)
    print(f"无环性: {'通过' if acyclic else '存在环路'}")

    # ---------- 层级深度 ----------
    depth_dict, max_depth, avg_depth, max_fanout, avg_fanout = compute_depth_and_fanout(ext_parent_of)
    print(f"最大层级 L_max: {max_depth}")
    print(f"平均层级 L_bar: {avg_depth:.4f}")

    # ---------- 分支广度 ----------
    print(f"最大扇出 f_max: {max_fanout}")
    print(f"平均扇出 f_bar: {avg_fanout:.4f}")

    # ---------- 知识覆盖 ----------
    # 概念覆盖率
    covered_concepts = ext_concepts.intersection(std_concepts)
    concept_coverage = len(covered_concepts) / len(std_concepts) if std_concepts else 0
    print(f"概念覆盖率 η_cov: {concept_coverage:.4f}")

    # 关系完整度（简化版：使用可还原的父子三元组比例，与边保留率相同，但论文中不同，此处输出占位符）
    # 由于论文中定义模糊，这里仅输出父子边保留率作为参考，实际应按论文要求定义
    relation_integraity = compute_relation_integrity(ext_data, emb_dict, std_parent_of, ext_parent_of, std_desc_of, std_concepts, 8)
    print(f"关系完整度 η_rel: 未明确定义，此处使用父子边保留率 {relation_integraity:.4f} 作为参考")

    # ---------- 语义一致 ----------
    # 需要使用CLIP嵌入（已提供）
    s_con, s_pair = compute_clip_similarities(ext_data, emb_dict)
    print(f"概念一致性 S_con: {s_con:.4f}")
    print(f"簇内相似度 S_pair: {s_pair:.4f}")

    # ---------- 语义多样 ----------
    d_lex, h_word = compute_lexical_diversity(ext_data)
    print(f"描述多样性得分 D_lex: {d_lex:.4f}")
    print(f"平均词汇熵 H_word: {h_word:.4f}")

    # ---------- 专家对齐 ----------
    # 语义相似度（BERT）
    sim_bert = compute_bert_similarity(ext_data, std_desc_of)
    if sim_bert is not None:
        print(f"语义相似度 Sim_BERT: {sim_bert:.4f}")
    else:
        print("语义相似度 Sim_BERT: BERT不可用，跳过计算")

    # 人工一致评分 和 幻觉率 需要人工评估，不计算
    print("\n注意：人工一致评分(Q_exp)和幻觉率(R_hallu)需人工标注，未在脚本中计算。")

if __name__ == "__main__":
    main()