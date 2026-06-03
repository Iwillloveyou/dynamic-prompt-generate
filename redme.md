
三.基于以上提示库的图文条件式动态提示网络
1.根据描述检索图像的条件式动态提示网络
训练网络脚本:dynamic_prompt_gennerate.py，训练完成后，模型存在checkpoints/best_generator.pth下。

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
