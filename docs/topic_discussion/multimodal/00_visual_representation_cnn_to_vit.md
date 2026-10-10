# 00 视觉表示：从 CNN 到 ViT

## 页面目标

图像进入检测模型或视觉语言模型前，必须先被转换为可计算的表示。CNN 通过局部卷积逐层扩大感受野；ViT 将图像拆成 patch token，再用注意力建立全局关系。两条路径的表示粒度、计算分布和输入长度不同，直接影响后续检测、视觉 token 与部署成本。

## 核心机制：局部特征与 patch token

| 机制 | CNN | ViT | 对后续任务的影响 |
|:---|:---|:---|:---|
| 基本单元 | 局部卷积核与特征图 | patch 与 token 序列 | 决定特征组织方式 |
| 空间关系 | 逐层扩大感受野 | 注意力直接关联 token | 影响长距离关系与计算分布 |
| 多尺度 | 特征金字塔或下采样层 | patch 层级、window 或额外特征层 | 影响小目标和不同分辨率 |
| 输入代价 | 像素大小影响 feature map | 分辨率与 patch size 决定 token 数 | 影响视觉编码时间、显存与 VLM Prefill |

![从图像网格到 CNN 特征图或 ViT patch token](../../public/topic_discussion/multimodal/cnn_vit_visual_representation.svg)

## 图像输入契约

| 字段 | 需要固定的内容 | 为什么需要记录 |
|:---|:---|:---|
| 图像 | 数量、尺寸、宽高比、颜色空间与格式 | 决定预处理和像素成本 |
| 预处理 | resize、crop、normalize、padding 与插值方式 | 同一图片可产生不同输入张量 |
| 表示 | backbone、patch size、输出层级与 dtype | 决定特征/token 数和精度路径 |
| 任务 | 分类、检测、VQA 或 OCR 的质量目标 | 防止只比较编码速度 |

## 连接到现有专题

| 需要补的能力 | 入口 | 本页如何使用 |
|:---|:---|:---|
| 张量、卷积与基础视觉计算 | [Part00 · 02 NumPy 与 einsum](../../00_Prerequisites/02_NumPy_and_Einsum.md)、[算子优化](../operator_optimization/intro.md) | 理解特征张量、访存和视觉编码热点 |
| Attention 与 Transformer Block | [Part02 · 04 Attention](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)、[05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md) | 将 patch token 连接到注意力计算 |
| 模型架构演进 | [大模型架构](../llm_architecture_evolution/intro.md) | 回补残差、归一化和 Transformer 设计选择 |

## 交付物

完成本页后，应能为一张图像写出“像素规模 → 特征图或 patch token → 下游模型输入”的表示账本，并说明分辨率或 patch size 改变时，哪些成本会随之变化。
