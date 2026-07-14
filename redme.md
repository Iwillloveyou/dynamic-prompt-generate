项目为基于ASAM OPENODD的动态提示学习的图文检索模型项目实现

一. OPENODD的提示库构建

对应文件在prompt_library下
ASAM OPENODD原始基础概念：openadd.json 其中openadd_desc.json是含有每个概念描述的
加载原始基础概念脚本：load_openadd.py
将原始基础概念转为概念树脚本：build_concept_tree.py，运行后得到：
ASAM OPENODD原始基础概念树：concept_tree.json
ASAM OPENODD基础概念树简化列表：concept_tree_simple.json
其中，
总节点数: 110
最大深度: 4
叶子节点数: 90
根节点: OperationalDesignDomain
✅ 树结构已保存至: concept_tree.json
✅ 简化列表已保存至: concept_tree_simple.json

📈 概念分布:
内部节点数: 20
叶子节点数: 90
最大深度: 4 (概念: Car)

针对概念树每个节点使用llm扩展描述脚本：llm-extend-doubao.py，其中，special_rules.json是针对特定部分概念在llm扩展时必须用到的微调规则；category_attributes.json针对部分概念的在llm扩展时必须用到的属性描述配置。当二者同时存在时，优先用special_rules中的。
产生的结果prompt_library/result下，文件说明：
concept.json 扩展的概念列表
concept_extend.json 概念扩展的结果列表，概念是name，其中name_emb是概念名的clip文本编码，desc_emb是多条场景描述的平均clip文本编码。示例如下：
{
"name": "OperationalDesignDomain",
"name_emb_key": "name_emb_0",
"desc": "The complete set of conditions under which the automated driving system is designed to operate.",
"desc_emb_key": "desc_emb_0",
"extend_desc": [
"A sunny afternoon on a multi-lane urban highway, with moderate traffic flow and clear lane markings, within the vehicle's operational design domain for highway driving.",
"Dense fog reduces visibility to 50 meters on a rural two-lane road at dawn, challenging the perception system's operational limits in adverse weather conditions.",
"Heavy rain at night on a well-lit city street with reflective road surfaces, testing the sensor suite's performance under combined low-light and precipitation conditions.",
"A complex urban intersection during evening rush hour with multiple traffic signals, pedestrian crossings, and cyclists, within the defined ODD for city navigation.",
"Dry asphalt on a clear mountain pass road with sharp curves and elevation changes, operating within the system's validated road geometry parameters.",
"Light snowfall during daytime in a residential area with parked vehicles on both sides, evaluating performance in winter conditions within specified temperature ranges.",
"A construction zone on a suburban arterial road with temporary lane markings and reduced speed limits, within the ODD for road work navigation.",
"A well-maintained parking garage with concrete pillars and artificial lighting at midday, operating within the confined space parameters of the automated parking system."
],
"extend_desc_emb_key": [
"extend_emb_0_0",
"extend_emb_0_1",
"extend_emb_0_2",
"extend_emb_0_3",
"extend_emb_0_4",
"extend_emb_0_5",
"extend_emb_0_6",
"extend_emb_0_7"
],
"desc_mean_emb_key": "desc_mean_emb_0"
}
concept_extend.embeddings.npz 存放了向量key及其embedding，后续参与动态提示生产就用到它们。

提示库指标评估时，需要安装相关包
pip install scikit-learn sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
其中nltk是必须的，可以执行nltk.install.python安装合验证

对比试验
主实验脚本: compare_experiment.py

基于cityflow-nl中描述高频词的提示库补充
先验计算发现有大量的nl描述词未出现在提示库中，因此通过聚类提取cityflow-nl中描述高频词，以进行扩充，主程序如下：
抽取高频词：cityflow_nl_category_extract.py
与当前openodd自动进行合并：auto_adapter_openodd.json.py

其中:
聚类提取全局和top800高频词，结果如high_freq_words_800.csv。
然后利用llm进行智能分类处理，与openodd种子概念进行聚合分类，得到concept_expand.json、concept_extend_expand.json。然后基于之前的llm进行提示扩展，生成新的提示库，提示库信息存在concept_extend_expand.embeddings.npz中。

二. 基于cityflow-nl的数据集准备
对应文件在cityflow_nl_dataset_handler下
训练的数据集来自cityflow-nl，需要提前参照AI City Challenge官方网站数据集（2023 Track 2: Tracked-Vehicle Retrieval by Natural Language Descriptions）和其官方GitHub仓库（https://github.com/Microsoft/CityFlow-NL）下载数据集与获取相关的预处理脚本和标注文件，并进行以下预处理：
1). 使用视频提取帧脚本将视频提取为图片
2). 解析其中包含的车辆轨迹图像以及对应的自然语言描述信息，建立自然语言查询到目标车辆的映射关系，即parse_annotations.py。
3). 按照7:2:1比例划分训练集、验证集、测试集，即split_dataset.py
处理脚本见prepare_cityflow_nl.sh。

预处理完成后，项目目录应如下所：
cityflow-nl/
├── images/                          # 所有提取的图像帧
│   ├── S01/
│   │   ├── c001/
│   │   │   ├── frame_000001.jpg
│   │   │   ├── frame_000002.jpg
│   │   │   └── ...
│   │   └── c002/
│   ├── S03/
│   └── S04/
├── annotations/                     # 处理后的标注文件
│   ├── cityflow-nl.json             # 原始合并标注
│   ├── train.json                    # 训练集标注
│   ├── val.json                      # 验证集标注
│   └── test.json                     # 测试集标注
├── data/                            # 原始数据（可选保留）
│   ├── train-tracks.json
│   ├── test-queries.json
│   └── train/
├── checkpoints/                     # 模型保存目录
└── prepare_cityflow_nl.sh           # 预处理脚本

另外，train_track_json_check.py 随机选择训练集中示cityflow-nl图片中的目标框进行显示以校验标注是否准确

三.基于以上提示库的图文条件式动态提示网络
1.根据描述（纯文本）检索图像的条件式动态提示网络
训练网络脚本:text_dynamic_prompt_gennerate.py，训练完成后，模型存在checkpoints/best_generator.pth下。

2.根据描述与参考图像联合检索的条件式动态提示网络
训练网络脚本:image_text_dynamic_prompt_gennerate.py

主实验部分
主程序image_text_dynamic_prompt_gennerate.py
为了保证使用同一训练和验证分类，将加载好的训练id放在checkpoints/track_split.pkl中，同时验证集候选特征放在candidate_feats_clipzeroshort.pt中

未优化的方法效率比较低，如下：

改进后的方法如下：
1.训练集构造时每个track_id每个nl不再只取1张图象，取5张以扩大训练集。同时放弃批内互斥采样，使用普通随机采样，并修改损失函数为多正样本对比损失，以避免同一批内采样到同一track_id多张匹配图片导致的原生对比损失假样本问题。
多正样本对比损失：multi_positive_contrastive_loss
2.修改学习率调度为scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
3.尝试新方法，将动态提示与图像特征融合，而非与文本特征融合

对比实验部分
1.原生clip模型
主程序：image_text_dynamic_prompt_prior.py
验证性能如下：
Recall@1: 1.55% Recall@5: 15.24% Recall@10: 24.75% mAP: 5.30%

同时，在此脚本中还增加了：
analyze_prior_stats方法：分析对提示库可以先验 prior 统计信息，使用验证集，直接计算对扩展描述、原始描述、概念名嵌入向量的得分均值，以衡量提示库的准确性
PriorOnlyMLPGenerator网络：直接将联合特征与所有概念特征相似度求权重，不是直接从原始特征预测权重的方法

2.clip4cir
主程序clip4cir_baseline.py
其中，需要预加载训练好的clip4cir模型，本文使用基于CIRR训练好的cirr_comb_RN50x4_fullft.pt模型，具体下载路径参照https://drive.google.com/drive/folders/1d3zayARGi7zG-wkojk-i0UBXrHe96CGT

预训练模型是基于CIRR训练的，不适配本文研究的自动驾驶垂直领域图文检索任务，训练性能如下：
Recall@1: 0.00% Recall@5: 0.08% Recall@10: 0.08% mr:0.05% mAP: 0.03%
因此需进行重新训练，训练方法见脚本define_train

直接在cityflow-nl上训练未优化的方法效率比较低(1epoch)，如下：
Recall@1: 0.00% Recall@5: 0.08% Recall@10: 0.08% mAP: 0.03%

改进后的方法如下：
1.训练集构造时每个track_id每个nl不再只取1张图象，取5张以扩大训练集。同时放弃批内互斥采样，使用普通随机采样，并修改损失函数为多正样本对比损失，以避免同一批内采样到同一track_id多张匹配图片导致的原生对比损失假样本问题。
多正样本对比损失：multi_positive_contrastive_loss
2.为了避免适配原特征512维到640维产生的特征破坏，修改combine网络的参数可传特征维度，并将adapter设为配置
2.修改学习率调度为scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


3.cocoop
主程序cocoop_baseline.py
其中，需要预加载训练好的coco及cocoop模型:
coop：本文使用基于CIRR训练好的cirr_comb_RN50x4_fullft.pt模型，具体下载路径参照https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view?usp=sharing，选取ViT-B/32的任意seed1
cocoop：暂未发现


预训练模型是基于ImageNet 训练的，不适配本文研究的自动驾驶垂直领域图文检索任务，训练性能如下：
Recall@1: 0.00% Recall@5: 0.00% Recall@10: 0.00% mAP: 0.02%
表现较低，因此需进行重新训练，训练方法见脚本define_train

直接在cityflow-nl上训练未优化的方法效率比较低(1epoch)，如下：
Recall@1: 0.08% Recall@5: 0.08% Recall@10: 0.08% mAP: 0.06%

改进后的方法如下：
1.训练集构造时每个track_id每个nl不再只取1张图象，取5张以扩大训练集。同时放弃批内互斥采样，使用普通随机采样，并修改损失函数为多正样本对比损失，以避免同一批内采样到同一track_id多张匹配图片导致的原生对比损失假样本问题。
多正样本对比损失：multi_positive_contrastive_loss
2.修改学习率调度为scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
3.将 CoCoOpTextEncoder 中的拼接改为替换，保持序列长度 77，并正确添加位置编码。
