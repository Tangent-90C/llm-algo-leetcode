# 多模态与视觉推理（Multimodal and Vision Inference）

> 专题类型：领域专题（建设中）　副标题：从视觉表示、实时检测到视觉语言模型与服务决策

## 页面导语

视觉推理不是把一张图交给语言模型那么简单。图像先经预处理和视觉表示，再进入检测模型或通过连接器成为语言模型上下文；不同任务会使用不同数据、质量指标、运行时和部署路径。本专题按“视觉表示 → 检测范式 → 实时视觉架构 → VLM → 服务优化”组织，让学习者能解释图像输入为何改变质量、延迟、显存与部署选择。

检测和 VLM 共享图像输入、视觉编码和部分运行环境，但不能共用一套结论：检测以 bbox、类别、AP/AR 为核心，VLM 以图像—文本任务质量、视觉 token 与服务指标为核心。每个 Task 都先固定数据、输入和环境，再比较候选模型或策略。

![多模态与视觉推理路线：从视觉表示到可复核服务](../../public/topic_discussion/multimodal/multimodal_overview.svg)

## 如何开始

- **希望理解视觉模型和实时检测：** Task0 → Task1 → Task2。
- **希望部署图像问答或视觉语言模型：** 从 Task3 开始；按需回补 Task0 的视觉表示和 Task1 的输入/评测契约。
- **希望优化服务：** 在固定 Task4 workload 后进入 Task5，避免用不同分辨率或输出长度比较后端。
- **希望做视频或视觉 Action：** 先完成图像主线，再进入 Task6 之后的时序扩展；视频帧数、采样率与时序标签会形成新的 workload。

## 主学习路线与验证出口

| Task / 主题 | 本阶段要回答的问题 | 机制与正文入口 | 主要交付物 |
|:---|:---|:---|:---|
| Task0 · 视觉表示：CNN 到 ViT | 图像如何变成局部特征或 patch token？ | [00 视觉表示](./00_visual_representation_cnn_to_vit.md)；CNN、ViT、多尺度表示、分辨率与 patch token | 像素 → 表示账本与输入契约 |
| Task1 · 检测范式：YOLO 到 DETR | 密集预测与 object query / 集合预测分别改变什么？ | [01 目标检测](./01_detection_yolo_to_detr.md)；标签、输出、NMS/匹配与 AP/AR | 检测输入/输出与评测契约 |
| Task2 · 实时检测 Transformer | RT-DETR 系列如何在训练、表示与部署路径上演进？ | [02 实时 DETR](./02_realtime_detr_evolution.md)；RT-DETR → v2 → v3 → v4 | 实时检测候选卡与质量—延迟证据 |
| Task3 · VLM 架构与视觉 token | 视觉特征如何通过连接器成为语言模型上下文？ | [03 VLM 架构与图像输入](./03_vlm_architecture_and_input_contract.md)；视觉塔、连接器、视觉 token、模板 | 图像—文本输入卡 |
| Task4 · 多模态 workload、缓存与 serving | 图像请求如何改变预处理、视觉编码、Prefill、Decode 和 KV？ | [04 多模态 Workload 与缓存](./04_vlm_workload_and_cache.md)；阶段拆分、视觉/文本状态与配对 baseline | 固定 workload、质量与性能结果表 |
| Task5 · 视觉/多模态 Profiling 与部署 | 慢在预处理、视觉编码、后处理、Prefill、Decode 还是 I/O？ | 关联：[性能优化](../performance_optimization/intro.md)、[部署与异构](../deployment_heterogeneous/intro.md)；正文待建 | 关键路径归因与 artifact/runtime 证据 |
| Task6 · 视觉与多模态综合项目 | 如何在质量、延迟、显存、设备与成本之间选择方案？ | 关联：[Part05 综合项目](../../05_Integrated_Projects_and_Decisions/intro.md)；后续建设检测部署与 VLM 服务项目 | 可复查报告与 `accept / tune / reject` |

## 两类 workload 与环境

| 路线 | 数据与质量 | 固定输入 | 典型性能字段 | 推荐环境 |
|:---|:---|:---|:---|:---|
| 实时检测 | 图像、bbox、类别；AP/AR、漏检/误检 | 尺寸、预处理、阈值、batch | 预处理、前向、后处理、P95、FPS、VRAM | CPU 机制 → 单 GPU PyTorch → export/runtime |
| VLM 服务 | 图像—问题—答案；VQA/OCR/grounding | 图片数、分辨率、文本、输出上限 | 视觉编码、Prefill、Decode、TTFT/TPOT、KV、VRAM | CPU 输入契约 → 单 GPU Transformers → backend |
| 视频 / 视觉 Action（扩展） | 帧序列、动作或事件标签；时序质量 | clip 长度、帧率、采样策略与音视频对齐 | 解码、帧采样、时序编码、端到端延迟 | 视频解码环境 + GPU；在图像主线稳定后建设 |

## 与其他专题的协作关系

| 专题 | 本专题复用什么 | 本专题补充什么 |
|:---|:---|:---|
| [数据工程](../data_engineering/intro.md) | 图像来源、标注/媒体解析、版本和评测隔离 | 检测与 VLM 的视觉数据契约 |
| [模型架构](../llm_architecture_evolution/intro.md) | Transformer、Attention、Block 设计 | CNN/ViT 视觉表示、检测 query 与视觉—语言连接 |
| [推理优化](../inference_optimization/intro.md) | Prefill、Decode、KV、服务指标 | 视觉 token 与媒体输入导致的工作负载变化 |
| [性能优化](../performance_optimization/intro.md) | benchmark、trace、I/O 与成本证据 | 预处理、视觉编码、后处理的阶段归因 |
| [部署与异构](../deployment_heterogeneous/intro.md) | artifact、runtime、硬件与部署契约 | 检测/VLM 的 processor、视觉塔和导出路径 |

## 相关阅读

- [RT-DETR / RT-DETRv2 官方实现](https://github.com/thaianh-pham/RT-DETRv2)
- [RT-DETRv3 官方实现](https://github.com/clxia12/RT-DETRv3)
- [RT-DETRv4 官方实现](https://github.com/RT-DETRs/RT-DETRv4)
- [vLLM 多模态输入](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)

以上实现和论文用于理解模型与运行时路径；不同数据、硬件、导出格式或 backend 的公开数字不能代替本专题固定 workload 下的复测。

## 建设状态

Task0–4 已提供路线与概念正文；Task5 将补齐视觉/多模态关键路径与 runtime 证据，Task6 再建设实时检测部署和 VLM 服务两个独立项目。视频与视觉 Action 作为图像主线后的扩展，不与当前检测或 VLM 结果表混合。
