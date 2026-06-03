# expand_concept_library.py
import csv
import json
import os
import re
from collections import Counter

import clip
import nltk
import numpy as np
import openai
import torch
from nltk.util import ngrams
from tqdm import tqdm

# ---------- 配置 ----------
# 路径配置
TRAIN_TRACKS_PATH = "../../dataset/cityflow-nl/train_tracks.json"
OPENODD_PATH = "openadd.json"  # 原始概念树文件
OUTPUT_JSON = "concept_extend.json"
OUTPUT_NPZ = "concept_extend.embeddings.npz"
STOPWORDS_CSV = "high_freq_words_800.csv"   # 输出高频词供人工筛选
SELECTED_CSV = "selected_concepts.csv"  # 人工筛选后的概念（需手动填写）

# LLM 配置
OPENAI_API_KEY = "your-api-key-here"   # 请替换或设置环境变量
openai.api_key = OPENAI_API_KEY
USE_LLM = True  # 若为 False，则使用模板生成扩展描述

# CLIP 配置
CLIP_MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. 提取高频词 ----------
def download_nltk_resources():
    import ssl
    import nltk
    ssl._create_default_https_context = ssl._create_unverified_context

    # 兼容写法
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt_tab')

    # stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def extract_high_freq_words(top_k=800):
    download_nltk_resources()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # 补充自定义停用词
    extra_stops = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'at', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'that', 'then', 'these', 'those', 'through', 'until', 'very', 'just', 'but', 'also', 'can', 'may', 'will', 'would', 'could', 'should', 'might'}
    stop_words.update(extra_stops)

    # 加载数据
    with open(TRAIN_TRACKS_PATH, 'r') as f:
        tracks = json.load(f)

    word_counter = Counter()
    for track_id, info in tracks.items():
        descriptions = info.get('nl', [])
        for desc in descriptions:
            # 分词并过滤
            words = nltk.word_tokenize(desc.lower())
            # 只保留字母词汇，且长度>1，且非停用词
            words = [w for w in words if w.isalpha() and len(w) > 1 and w not in stop_words]
            word_counter.update(words)

    # 输出前500个高频词到CSV
    most_common = word_counter.most_common(top_k)
    with open(STOPWORDS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word', 'frequency', 'category', 'parent_node', 'notes'])
        for word, freq in most_common:
            writer.writerow([word, freq, '', '', ''])

    print(f"High frequency words saved to {STOPWORDS_CSV}. Please manually edit this file, fill in 'category' (noun/verb/adj) and 'parent_node' (where to insert in OpenODD tree), then save as {SELECTED_CSV}.")

def extract_high_freq_phase(ngram_range=(1, 3), top_k=800):
    download_nltk_resources()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # 补充自定义停用词（同上）
    extra_stops = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'at', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'that', 'then', 'these', 'those', 'through', 'until', 'very', 'just', 'but', 'also', 'can', 'may', 'will', 'would', 'could', 'should', 'might'}
    stop_words.update(extra_stops)

    with open(TRAIN_TRACKS_PATH, 'r') as f:
        tracks = json.load(f)

    # 存储所有 n-gram 及其频率
    ngram_counter = Counter()

    for track_id, info in tracks.items():
        # 合并 nl 和 nl_other_views
        descriptions = info.get('nl', []) + info.get('nl_other_views', [])
        for desc in descriptions:
            # 清洗：转小写，去掉标点（保留字母和空格）
            cleaned = re.sub(r'[^a-z\s]', '', desc.lower())
            tokens = cleaned.split()
            # 过滤短词和停用词（但对短语不直接过滤，因为短语中的词可能含停用词）
            # 注意：对于短语，我们保留原始序列但每个词仍可被停用词过滤会导致短语断裂，故先不断裂
            # 方法：生成 n-gram 时，如果 n-gram 中包含停用词，仍然保留，因为短语意义重要（如 "turn right" 中 "right" 不是停用词）
            # 但为了降低噪声，可以过滤长度 <=1 的 token
            if not tokens:
                continue
            # 生成双词短语
        # 单字词：遍历所有 token
        for w in tokens:
            if len(w) > 2 and w not in stop_words:
                ngram_counter[w] += 1

        # 双词短语：遍历相邻对
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if (len(w1) > 2 and w1 not in stop_words and len(w2) > 2 and w2 not in stop_words):
                phrase = f"{w1} {w2}"
                ngram_counter[phrase] += 1
            # # 生成 n-gram
            # for n in range(ngram_range[0], ngram_range[1]+1):
            #     for gram in ngrams(tokens, n):
            #         phrase = ' '.join(gram)
            #         # 可选：过滤明显无意义的短语（如全是停用词，或长度太短）
            #         if len(phrase) < 3:
            #             continue
            #         ngram_counter[phrase] += 1

    # 输出 top_k 高频短语
    most_common = ngram_counter.most_common(top_k)
    with open(STOPWORDS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['phrase', 'frequency', 'category', 'parent_node', 'notes'])
        for phrase, freq in most_common:
            writer.writerow([phrase, freq, '', '', ''])

    print(f"High frequency phrases (n-gram) saved to {STOPWORDS_CSV}. Please manually edit this file, fill in 'category' and 'parent_node' for each phrase you want to include, then save as {SELECTED_CSV}.")
# ---------- 2. 加载原始概念树 ----------
def load_original_tree(path):
    with open(path, 'r') as f:
        return json.load(f)

def find_node_by_name(tree, node_name, parent=None):
    """在树中递归查找节点名称，返回节点字典及其父节点"""
    if tree.get('name') == node_name:
        return tree, parent
    for child_name, child_node in tree.items():
        if child_name == node_name:
            return child_node, tree
        if isinstance(child_node, dict):
            found, par = find_node_by_name(child_node, node_name, child_node)
            if found:
                return found, par
    return None, None

def add_concept_to_tree(tree, concept_name, parent_node_name):
    """
    将新概念添加到指定父节点下。如果父节点不存在，则在根节点下创建新的分支。
    tree: 原始树字典（根为 OperationalDesignDomain）
    返回修改后的树
    """
    # 如果父节点为根节点或未指定，则直接添加到根下
    if parent_node_name == 'OperationalDesignDomain' or not parent_node_name:
        if concept_name not in tree['OperationalDesignDomain']['children']:
            tree['OperationalDesignDomain']['children'].append(concept_name)
            tree[concept_name] = {'parent': 'OperationalDesignDomain', 'children': []}
        return tree
    # 查找父节点
    parent_node, _ = find_node_by_name(tree, parent_node_name)
    if parent_node is None:
        # 父节点不存在，创建新的父节点在根下？或者忽略
        print(f"Warning: parent node {parent_node_name} not found, adding {concept_name} under root.")
        if concept_name not in tree['OperationalDesignDomain']['children']:
            tree['OperationalDesignDomain']['children'].append(concept_name)
            tree[concept_name] = {'parent': 'OperationalDesignDomain', 'children': []}
    else:
        if 'children' not in parent_node:
            parent_node['children'] = []
        if concept_name not in parent_node['children']:
            parent_node['children'].append(concept_name)
            tree[concept_name] = {'parent': parent_node_name, 'children': []}
    return tree

# ---------- 3. 生成扩展描述 ----------
def generate_extended_descriptions(concept_name, category, num=5):
    """
    调用 LLM 生成该概念的多条扩展描述（适合 CLIP 编码的场景描述）。
    如果 USE_LLM=False，则使用模板生成简单描述。
    """
    if USE_LLM and OPENAI_API_KEY != "your-api-key-here":
        prompt = f"Generate {num} short, diverse descriptions (each 5-15 words) for the driving scene concept '{concept_name}' (category: {category}). The descriptions should be realistic and varied, covering different scenarios. Return each description on a new line, without numbering."
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=200
            )
            desc_text = response.choices[0].message.content.strip()
            descriptions = [line.strip() for line in desc_text.split('\n') if line.strip()]
            if len(descriptions) < num:
                descriptions += [f"A typical driving scenario involving {concept_name}."] * (num - len(descriptions))
            return descriptions[:num]
        except Exception as e:
            print(f"LLM error for {concept_name}: {e}")
            # fallback to template
    # 模板生成
    templates = [
        f"Driving scene with {concept_name}.",
        f"Vehicle exhibits {concept_name}.",
        f"The road condition shows {concept_name}.",
        f"Traffic participants demonstrate {concept_name}.",
        f"Observing {concept_name} in autonomous driving context."
    ]
    return templates[:num]

# ---------- 4. 主流程：读取筛选后的 CSV，构建新概念库 ----------
def build_new_concept_library():
    # 检查人工筛选文件是否存在
    if not os.path.exists(SELECTED_CSV):
        print(f"Please edit {STOPWORDS_CSV} and save as {SELECTED_CSV} with columns: word, frequency, category, parent_node, notes")
        return

    # 加载原始树
    tree = load_original_tree(OPENODD_PATH)
    # 确保根节点为 OperationalDesignDomain
    if 'OperationalDesignDomain' not in tree:
        tree = {'OperationalDesignDomain': tree}  # 适配格式

    # 读取筛选后的概念
    new_concepts = []
    with open(SELECTED_CSV, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            word = row['word'].strip()
            if not word:
                continue
            category = row.get('category', '').strip()
            parent = row.get('parent_node', '').strip()
            if not parent:
                parent = 'OperationalDesignDomain'
            new_concepts.append({'name': word, 'category': category, 'parent': parent})

    # 将新概念插入树中
    for concept in new_concepts:
        add_concept_to_tree(tree, concept['name'], concept['parent'])

    # 生成最终的概念列表（包含原始节点和新节点）
    # 遍历树，收集所有节点名称、父节点、扩展描述
    concept_list = []
    def traverse(node_dict, node_name, parent):
        # 跳过根节点
        if node_name != 'OperationalDesignDomain':
            concept_list.append({
                'name': node_name,
                'parent': parent,
                'category': '',  # 原始节点 category 留空
                'extend_desc': []  # 稍后生成
            })
        children = node_dict.get('children', [])
        for child_name in children:
            if child_name in node_dict:
                traverse(node_dict[child_name], child_name, node_name)

    # 从根开始遍历
    traverse(tree['OperationalDesignDomain'], 'OperationalDesignDomain', None)

    # 为每个概念生成扩展描述（只针对新概念，也可以全部重新生成）
    # 这里我们统一为所有概念生成，但原始概念可保留已有扩展（如果有）
    # 为了简单，我们为新概念生成扩展描述，原始概念的扩展描述可以暂时留空或使用已有
    # 最终输出 JSON 时，每个概念包含 name, parent, extend_desc, desc_mean_emb 等字段，但 embedding 稍后计算
    # 我们先构建概念列表，并生成扩展描述
    existing_names = set()
    # 加载原有的 concept_extend.json（如果存在）以保留旧扩展？
    # 这里假设我们全新生成
    for concept in concept_list:
        # 对于新概念（出现在 selected CSV 中），生成扩展描述
        if concept['name'] in [c['name'] for c in new_concepts]:
            descs = generate_extended_descriptions(concept['name'], concept['category'], num=5)
            concept['extend_desc'] = descs
            concept['new'] = True
        else:
            concept['extend_desc'] = []
            concept['new'] = False

    # 保存为 concept_extend.json（不含 embedding）
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        # 为了保持与原来格式兼容，存储为列表，每个 item 包含 name, parent, extend_desc 等
        # 注意原来的格式可能包含 name_emb_key, extend_desc_emb_key 等，这些在后续生成 npz 时添加
        output_data = []
        for idx, c in enumerate(concept_list):
            item = {
                'name': c['name'],
                'parent': c['parent'],
                'category': c.get('category', ''),
                'extend_desc': c['extend_desc'],
                'name_emb_key': f"name_emb_{idx}",
                'extend_desc_emb_key': [f"extend_emb_{idx}_{j}" for j in range(len(c['extend_desc']))],
                'desc_mean_emb_key': f"desc_mean_emb_{idx}"
            }
            output_data.append(item)
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Concept JSON saved to {OUTPUT_JSON}. Now generating embeddings with CLIP...")
    return output_data

# ---------- 5. 生成 CLIP embedding 并保存为 npz ----------
def generate_embeddings(concept_json_path, npz_path):
    # 加载 CLIP
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    with open(concept_json_path, 'r') as f:
        concepts = json.load(f)

    npz_data = {}
    for item in tqdm(concepts, desc="Encoding concepts"):
        name = item['name']
        # 编码名称
        text_token = clip.tokenize([name], truncate=True).to(DEVICE)
        with torch.no_grad():
            name_emb = clip_model.encode_text(text_token).cpu().numpy().squeeze()
        name_emb = name_emb / np.linalg.norm(name_emb)  # 归一化
        npz_data[item['name_emb_key']] = name_emb

        # 编码扩展描述
        extend_embs = []
        for desc in item['extend_desc']:
            desc_token = clip.tokenize([desc], truncate=True).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_text(desc_token).cpu().numpy().squeeze()
            emb = emb / np.linalg.norm(emb)
            extend_embs.append(emb)
        # 存储每个扩展 embedding
        for key, emb in zip(item['extend_desc_emb_key'], extend_embs):
            npz_data[key] = emb
        # 计算扩展描述的均值 embedding
        if extend_embs:
            mean_emb = np.mean(extend_embs, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
        else:
            mean_emb = name_emb  # 如果没有扩展描述，使用名称 embedding
        npz_data[item['desc_mean_emb_key']] = mean_emb

    # 保存 npz
    np.savez(npz_path, **npz_data)
    print(f"Embeddings saved to {npz_path}")

# ---------- 入口 ----------
if __name__ == "__main__":
    # 第一步：提取高频词（若之前未运行过，请先运行以下行；若已有 CSV 可跳过）
    if not os.path.exists(STOPWORDS_CSV):
        extract_high_freq_phase(top_k=800)
        print(f"请手动编辑 {STOPWORDS_CSV}，填写 category 和 parent_node，然后保存为 {SELECTED_CSV}，再重新运行此脚本。")
        exit(0)

    # # 第二步：构建新概念库（需人工筛选文件已存在）
    # if not os.path.exists(SELECTED_CSV):
    #     print(f"未找到 {SELECTED_CSV}，请先根据 {STOPWORDS_CSV} 进行人工筛选。")
    #     exit(0)
    #
    # concepts_data = build_new_concept_library()
    # # 第三步：生成 embedding
    # generate_embeddings(OUTPUT_JSON, OUTPUT_NPZ)
    # print("Done.")