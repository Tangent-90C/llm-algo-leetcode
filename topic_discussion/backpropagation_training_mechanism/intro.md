# 反向传播与训练机制（Backpropagation and Training Mechanics）

> 专题类型：基础支撑　副标题：从梯度信号、训练状态到正确性与性能证据

一次训练更新先由前向传播建立依赖，再由损失给出监督信号，最后由反向传播生成梯度并写回参数。参数、激活值、反向保存张量、梯度和优化器状态在不同阶段产生、驻留和释放；这些关系共同决定更新是否正确、峰值显存出现在哪里，以及重算或搬运会增加多少代价。

本专题维护预训练、持续预训练和后训练共用的训练机制，不单独偏向其中一条路线。不同阶段会改变数据来源、监督范围、参与模型和分布式规模，但都会复用计算图、局部反向、有效更新、数值稳定和训练状态生命周期。学习者完成核心任务后，应能画出训练依赖图，核对监督与更新口径，建立状态生命周期账本，并在比较性能之前验证梯度和数值行为。

## 学习路线

按 Task 顺序学习时使用下表；需要连续理解一次训练更新时进入[训练机制问题链](./walkthrough.md)，遇到梯度为空、更新不一致或显存峰值异常时使用[训练机制问题索引](./casebook.md)，术语差异查询[专题术语表](./glossary.md)。Task0–4 建立共用训练机制，Task5 比较重算与搬运，Task6 将正确性、数值和性能证据收口为采用决策。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 训练闭环与梯度信号 | 参数如何经过前向传播、损失、反向传播和更新形成一次完整训练闭环？ | 核心：[Part 00 · 09 模块与参数](../../00_Prerequisites/09_PyTorch_nn_Module_Basics.ipynb) → [Part 00 · 12 训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb) | 参数初态 → 前向依赖 → 损失 → 梯度 → 更新后状态 | [01 训练闭环与计算图](./01_training_loop_and_computation_graph.md) |
| Task1 · 计算图、VJP 与梯度边界 | 自动求导如何沿计算图传播上游梯度，计算图在何处被保留、累积或切断？ | 核心：[Part 00 · 07 自动求导](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb) → [Part 00 · 08 梯度边界](../../00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad.ipynb) | 局部导数 → VJP → 分支汇合 → 累积 / 切断 | [02 自动求导与 Transformer 反向传播](./02_autograd_and_transformer_backward.md) |
| Task2 · Transformer 局部反向与保存状态 | 激活、损失、Attention 与残差路径分别依赖哪些前向状态？ | 核心：[Part 02 · 09 激活与损失反向](../../02_PyTorch_Algorithms/09_Activation_and_Loss_Backward.ipynb) → [Part 02 · 10 Attention 反向传播](../../02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.ipynb) | 局部算子 → 保存状态 → 反向消费者 → 梯度对照 | [02 自动求导与 Transformer 反向传播](./02_autograd_and_transformer_backward.md) |
| Task3 · 监督、梯度尺度与有效更新 | 损失归约、微批次、混合精度与更新时钟如何共同决定一次有效更新？ | 基础：[Part 01 · 12 Tensor Core 与混合精度](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)；核心：[Part 02 · 11 梯度累积](../../02_PyTorch_Algorithms/11_Gradient_Accumulation.ipynb) → [Part 02 · TRAIN-MIXED-PRECISION-BACKWARD 混合精度反向](../../02_PyTorch_Algorithms/TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb) → [Part 02 · 12 优化器与调度](../../02_PyTorch_Algorithms/12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb)；扩展：[Part 02 · 13 SFT 监督范围](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.ipynb) | 监督范围 → 损失分母 → 梯度累积 / 缩放 → 有效更新 | [03 监督与有效更新](./03_supervision_gradient_scale_effective_update.md) |
| Task4 · 训练状态生命周期与峰值账本 | 参数、梯度、优化器状态、激活和工作区在何时产生、驻留、消费与释放？ | 核心：[Part 00 · 18 显存账本](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb) → [Part 02 · 43 统一训练内存账本](../../02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.ipynb) | 状态分类 → 生命周期 → 阶段重叠 → 峰值贡献 | [04 训练状态生命周期](./04_training_state_lifecycle.md) |
| Task5 · 重算、搬运与驻留策略 | 激活检查点和卸载分别改变什么状态，代价由计算还是传输承担？ | 核心：[Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.ipynb) → [Part 02 · 42 激活卸载](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb)；项目：[Part 02 · 76 策略对照](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) | 作用对象 → 正确性 → 容量收益 → 重算 / 搬运代价 | [05 激活检查点与卸载](./05_checkpointing_and_offload.md) |
| Task6 · 正确性、数值稳定与性能证据 | 梯度和更新语义成立后，如何判断候选策略既正确又值得采用？ | 核心：[Part 00 · 19 异常定位](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb) → [Part 00 · 20 实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb)；项目：[Part 02 · 73 训练基线](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [Part 02 · 74 性能归因](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) → [Part 02 · 76 策略对照](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) | 复用 Task3 数值证据 → 梯度正确 → 质量过门 → 性能复测 → 决策 | [06 正确性与性能证据](./06_correctness_and_performance_evidence.md) |

![反向传播与训练机制学习路线](../../docs/public/topic_discussion/backpropagation_training_mechanism/backprop_training_roadmap.svg)

![反向传播与训练机制知识地图](../../docs/public/topic_discussion/backpropagation_training_mechanism/backprop_training_knowledge_map.svg)

## 扩展与跨专题入口

### 共同机制的应用路径

共同主干分别服务两条应用路径：预训练与持续预训练使用全序列监督和长周期训练，后训练引入 response-only mask、adapter、偏好对、rollout 与奖励信号。两条路径改变监督与系统规模，不改变计算图、有效更新和状态生命周期的基本语义。

| 应用路径 | 主要变化 | 复用的共同机制 |
| --- | --- | --- |
| 预训练 / 持续预训练 | 全序列监督、长周期 checkpoint、较大的数据与并行规模 | 计算图、局部反向、梯度累积、混合精度、状态生命周期 |
| 后训练 | response-only mask、adapter、偏好对、rollout、reward / value | 有效更新、有限值检查、梯度尺度、检查点与卸载、正确性证据 |

### 按需回补

| 知识缺口 | 回补入口 | 使用时机 |
| --- | --- | --- |
| Attention 的前向表示 | [Part 02 · 04 Attention](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) | Task2 中需要回看 Q/K/V、Softmax 与残差路径的输入输出时 |
| 参数更新与学习率节奏 | [Part 02 · 12 优化器与调度](../../02_PyTorch_Algorithms/12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb) | Task3 中需要核对学习率调度器与优化器更新步时 |
| dtype、FP16 与 BF16 | [Part 01 · 01 数据格式与混合精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) | Task6 中分析梯度范围、缩放和有限值时 |
| ZeRO 与多卡状态切分 | [Part 01 · 06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) | Task4 账本扩展到多卡训练时 |
| 梯度同步与通信 | [通信与并行](../communication_parallelism/intro.md) | 单卡梯度语义已经成立，需要分析 All-Reduce 或 Reduce-Scatter 时 |

### 跨专题入口

| 后续问题 | 进入专题 | 本专题提供的基础 |
| --- | --- | --- |
| SFT、DPO、GRPO 的训练目标和项目交付 | [后训练优化](../post_training_optimization/intro.md) | 监督信号、有效更新和梯度安全 |
| 峰值显存、时间线追踪、吞吐和策略选择 | [性能优化](../performance_optimization/intro.md) | 状态生命周期、策略作用对象和正确性门槛 |
| 反向传播内核、融合与重算 | [算子优化](../operator_optimization/intro.md) | 局部梯度依赖与反向保存张量 |
| 状态切分、梯度同步和扩展效率 | [通信与并行](../communication_parallelism/intro.md) | 单卡梯度语义与训练状态账本 |

## 实验与项目出口

机制页先输出梯度、保存状态、有限值和更新边界；项目页再验证训练基线、策略归因与组合决策。稳定出口分别是 `TRAIN-BASELINE`（Part 02 · 73）、`TRAIN-PROFILING-DIAGNOSIS`（Part 02 · 74）和 `TRAIN-MEMORY-STRATEGY`（Part 02 · 76）。

### 环境与验证

| 实验层级 | 适用内容 | 可确认的结论 |
| --- | --- | --- |
| CPU 机制 | Task0–5 的计算图、局部梯度、监督、账本和策略逻辑 | 数值关系、状态转移与输入契约正确 |
| 单卡 GPU | 梯度累积、混合精度、激活检查点与卸载 | 更新一致性、峰值显存与单步时间的真实代价 |
| 训练性能项目 | Part 02 · 73、74、76 | 固定工作负载下候选策略是否值得采用 |

单卡机制实验统一使用 `TRAIN-CUDA-MECHANISM`。Notebook 的 5.1 只保留 `RUN_MODE`、`RUN_LABEL`、Profile ID 和本页工作负载参数；安装命令、依赖互斥与故障处理集中在[实验 Quickstart](./quickstart.md)，页面与结果状态集中在[实验资产表](./experiment_assets.md)。

项目实验固定模型、数据划分、有效批次、数据类型、优化器更新次数和随机种子。结果字段、Profile、页面状态和证据等级统一登记在[实验资产表](./experiment_assets.md)。
