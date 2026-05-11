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


三.基于以上提示库的图文条件式动态提示网络
1.根据描述检索图像的条件式动态提示网络
训练网络脚本:dynamic_prompt_gennerate.py，训练完成后，模型存在checkpoints/best_generator.pth下。

2.根据描述与参考图像联合检索的条件式动态提示网络
训练网络脚本:image_text_dynamic_prompt_gennerate.py

