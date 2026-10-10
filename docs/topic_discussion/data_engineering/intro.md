# 数据工程（Data Engineering）

> 专题类型：横向工程专题（建设中）　主服务目标：把可追溯的数据生命周期接到训练、RAG 与性能证据

## 页面导语

数据工程不是后训练的附属步骤。训练数据、RAG 语料与线上知识库都需要经历采集、解析、清洗、去重、质量评估、版本治理和格式化；它们只在尾部进入不同消费者：微调进入样本配比与训练，RAG 进入切块、富化与向量索引，性能工程则观察数据加载、token 长度和 I/O 对训练吞吐的影响。

本专题把这些共用环节组织成一条横向能力线。后训练优化、性能优化和推理优化分别负责其领域的训练决策、测量证据和服务决策；数据工程负责保证输入数据、血缘与质量判断可复查。

## 如何开始

按知识顺序学习时，从 Task0 开始，先建立数据对象、版本和消费者的共同口径；再按目标进入对应分支。

- **建设 SFT 或偏好数据：** Task0 → Task2 → Task3 → Task5 → Task6。
- **建设 RAG 语料：** Task0 → Task1 → Task2；完成 chunk、富化与索引后转入[推理优化](../inference_optimization/intro.md)。
- **排查训练吞吐或长序列预算：** Task0 → Task2 → Task3，并将长度、packing、预处理和 I/O 记录带入[性能优化](../performance_optimization/intro.md)。
- **建设合成或 Agent 轨迹数据：** Task0 → Task4 → Task6；任务级执行反馈再进入[Agent Infra](../agent/intro.md)。

## 主学习路线与验证出口

下表按“要回答的问题 → 学习入口 → 验证出口”组织。现有 Notebook 优先承担可运行的机制和项目验证；尚未建设的解析、合成与编排内容先由专题正文明确问题、数据契约和交付物，避免把规划误写成已有实验结论。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序与正文入口 |
|:---|:---|:---|:---|
| Task0 · 数据生命周期与消费者 | 一份数据如何从来源、版本、处理规则流向预训练、中期训练、后训练和 RAG？ | 共享前置：[Part00 · 04 配置与数据入口](../../00_Prerequisites/04_Python_Config_and_Data_Entry.md)；核心：[32 SFT 数据准入与长度审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md)；正文：[00 数据生命周期与消费者契约](./00_data_lifecycle_and_contract.md) | 先建立数据来源—处理—消费者地图，再确定数据契约 |
| Task1 · 采集、解析与数据契约 | 异构文件如何按内容检测、解析和路由，并保留来源、许可和处理记录？ | 共享前置：[Part00 · 04](../../00_Prerequisites/04_Python_Config_and_Data_Entry.md)；关联：[部署与异构](../deployment_heterogeneous/intro.md)；正文：[01 采集、解析与来源契约](./01_collection_parsing_contract.md) | 先完成统一结构化文本，再进入清洗 |
| Task2 · 清洗、去重与质量过滤 | 如何识别空值、乱码、重复、污染与低质量文本，并用下游证据验证规则？ | [32 数据工程](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) → [50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) → [64 数据质量项目](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md)；正文：[02 清洗、去重与质量证据链](./02_cleaning_quality_chain.md) | 字段契约 → 精确/近似去重 → 质量与污染检查 → 数据准入 |
| Task3 · 数据配比与课程学习 | 来源、长度、难度与长尾样本如何配比，并用消融确认收益？ | [30 长上下文微调机制](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md) → [32 数据准入与长度审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) → [33 微调准入与最小干预](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md)；正文：[03 数据配比与课程学习](./03_data_mixture_and_curriculum.md) | 先看长度与预算，再设计来源/难度配比，最后进入训练复测 |
| Task4 · 合成数据与验证 | 如何生成稀缺场景样本，并验证任务、回答或执行轨迹质量？ | [50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) → [51 在线反馈](../../02_PyTorch_Algorithms/51_Online_DPO.md)；关联：[后训练优化](../post_training_optimization/intro.md)、[Agent Infra](../agent/intro.md) | 任务蓝图 → 多候选生成 → 规则/模型/执行验证 → 数据版本；正文待建：合成数据与验证 |
| Task5 · 后训练数据工程 | SFT、偏好与 CoT 数据如何格式化、去污染、接入训练框架并诊断“假学会”？ | [13 SFT 训练循环](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) → [15 端到端微调](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.md) → [32 数据工程](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) → [50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md)；项目：[62](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)、[64](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md)、[84–86](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md)；正文：[05 后训练数据接口与诊断](./05_post_training_data_interfaces.md) | SFT 数据契约 → 偏好/CoT 数据 → 切分与评测 → 训练产物 |
| Task6 · 流水线工程化与综合项目 | 如何让处理过程可复现、可增量运行、可审计，并验证数据改变的下游影响？ | [64 数据质量项目](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) → [Part05 综合项目](../../05_Integrated_Projects_and_Decisions/intro.md)；后续接入 DVC / Dagster / OpenLineage | 固定版本与规则 → 编排处理 → 记录血缘 → 下游训练/RAG 验证；正文待建：流水线工程化与项目决策 |

## 按需回补与跨专题出口

| 主线 | 数据工程提供的输入与证据 | 专题侧的最终判断 |
|:---|:---|:---|
| 后训练优化 | SFT / 偏好数据契约、去重、污染检查、配比与数据版本 | 模型质量、对齐和训练交付是否可信 |
| 性能优化 | token 长度、packing、数据加载与预处理时间、I/O 等待 | 训练吞吐或端到端成本的瓶颈归因 |
| 推理优化 | 文档解析、清洗、切块、富化与索引输入 | RAG 请求质量、检索成本与服务质量 |

| 需要补的能力 | 当前入口 | 建议时机 |
|:---|:---|:---|
| 配置、路径与临时文件 | [Part00 · 04](../../00_Prerequisites/04_Python_Config_and_Data_Entry.md) | Task0 / Task1 前 | 
| tokenizer、labels 与训练接口 | [13 SFT 训练循环](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) → [15 端到端微调](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.md) | Task5 前 |
| 长度、packing 与训练预算 | [30 长上下文微调机制](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md) → [33 微调准入与最小干预](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md) | Task3 / Task5 |
| 数据质量与评测隔离 | [50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) → [64 数据质量项目](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) | Task2、Task5、Task6 |
| 训练侧数据加载与 I/O 证据 | [性能优化](../performance_optimization/intro.md) | 发现吞吐或等待异常时 |
| RAG 消费者与服务证据 | [推理优化](../inference_optimization/intro.md) | Task1 / Task2 完成后 |

## 连续阅读与问题判断

- 想沿一份真实数据连续推进：阅读[数据工程问题链](./walkthrough.md)。
- 已有数据集、训练结果或 RAG 失败样本：阅读[数据工程判断手册](./casebook.md)。

## 环境、证据与建设顺序

| 实验类型 | 环境 | 可以确认的结论 |
|:---|:---|:---|
| CPU 机制 | Python / PyTorch / tokenizer mock | 字段、解析路由、去重、切分、长度与规则逻辑 |
| 单卡训练 | Transformers / Datasets / PEFT | packing、DataLoader、训练数据格式与下游质量变化 |
| RAG / Serving | 检索组件 + 推理 backend | 文档处理对召回、延迟、成本和失败样本的影响 |
| 工程化项目 | 版本库 + 编排器 + 观测工具 | 数据版本、血缘、增量运行、告警与可复现交付 |

Task0–3、Task5 已把 32、50、64 收束为可复用的数据契约、质量证据、数据配方与后训练接口；下一阶段建设 Task4、Task6 的合成与编排项目。没有固定数据版本、评测集隔离和下游验证时，任何“数据更好”的结论都只能视为候选，不应当作训练或 RAG 的最终结论。
