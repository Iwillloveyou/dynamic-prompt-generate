import json
import numpy as np

RESULT_SAVE_DIR = "./result/"
OUTPUT_VECTORS = RESULT_SAVE_DIR + "concept_vectors.npy"
CONCEPT_EXTEND_OUTPUT_NAMES = RESULT_SAVE_DIR + "concept_extend.json"
CONCEPT_EXTEND_EMBEDDING_OUTPUT_NAMES= RESULT_SAVE_DIR + "concept_extend.embeddings.npz"

# 1. 读取文本数据
with open(CONCEPT_EXTEND_OUTPUT_NAMES, "r", encoding="utf-8") as f:
    text_data = json.load(f)

# 2. 读取嵌入向量
emb_data = np.load(CONCEPT_EXTEND_EMBEDDING_OUTPUT_NAMES)

# 3. 重组完整数据
final_data = []
for item in text_data:
    # 读取原始描述嵌入
    desc_emb = emb_data[item["desc_emb_key"]]
    # 读取扩展描述嵌入
    extend_desc_embs = [emb_data[key] for key in item["extend_desc_emb_key"]]

    final_data.append({
        "name": item["name"],
        "name_embdding": emb_data[item["name_emb_key"]],
        "desc": item["desc"],
        "desc_embdding": desc_emb,
        "extend_desc": item["extend_desc"],
        "extend_desc_embdding": extend_desc_embs,
        "desc_mean_emb": emb_data[item["desc_mean_emb_key"]],
    })

# 打印结果
print("重组后的完整数据：")
for data in final_data:
    print(f"姓名：{data['name']}")
    print(f"原始描述：{data['desc']}")
    print(f"原始描述嵌入形状：{data['desc_embdding'].shape}")
    print(f"扩展描述：{data['extend_desc']}")
    print(f"扩展描述嵌入数量：{len(data['extend_desc_embdding'])}")
    print(f"扩展描述平均嵌入形状：{data['desc_mean_emb'].shape}")
    print("-" * 50)