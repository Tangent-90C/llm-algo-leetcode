# 03 训练状态与后训练资源账本

## 页面目标

训练资源不是一个单独的“模型大小”数字。参数、梯度、optimizer state、activation、临时 workspace 与通信缓冲区会在不同阶段同时驻留；进入 SFT、DPO 或 RL 后，还会增加 adapter、reference、reward/verifier、rollout 结果和环境状态。优化策略的价值取决于它降低了哪类峰值，以及把代价转移到了计算、采样、搬运、通信还是质量上。

本页以账本组织通用训练和后训练资源问题。它不解释 DPO、PPO、GRPO 的目标函数，也不替代 checkpoint、offload 或 ZeRO 的机制页；它负责把这些 workload 转成可比较的驻留对象、阶段时间和质量门槛。

## 核心机制：训练峰值来自驻留重叠

| 对象 | 常见驻留阶段 | 主要影响因素 | 典型候选策略 |
|:---|:---|:---|:---|
| 参数与 master weights | 全训练过程 | 模型规模、dtype、是否复制 | 混合精度、量化、分片 |
| 梯度与 optimizer state | backward 后至 update | 参数规模、optimizer、world size | 累积、ZeRO / 分片、低精度状态 |
| activation | forward 到对应 backward | batch、序列长度、层数、attention | checkpoint、offload、重排计算 |
| workspace / 临时张量 | 特定 operator 或通信阶段 | kernel、shape、collective、并发 | 算子路径、布局、调度与 overlap |
| 通信缓冲区 | collective 与同步窗口 | 并行策略、拓扑、bucket | bucket、分片、通信—计算重叠 |

账本要记录对象的大小、创建点、峰值时是否与其他对象重叠，以及策略改变后新增的时间或通信代价。这样才能解释“显存下降但 step time 变慢”究竟由重算、搬运还是同步造成。

| 账本字段 | 为什么记录 | 常见误读 |
|:---|:---|:---|
| 对象与创建阶段 | 区分加载、forward、backward、update 中的峰值来源 | 把所有显存都归因于模型参数 |
| allocated / reserved / 可回收空间 | 区分真实活跃张量与 allocator 保留 | 将 reserved 直接当作泄漏或策略收益 |
| saved tensors / activation 区段 | 确认哪些前向结果必须等到反向使用 | 只按层数估算而忽略序列长度和保存策略 |
| workspace / communication buffer | 解释短时峰值和多卡额外占用 | 在参数账本中漏掉临时执行空间 |
| 策略后的重算、搬运或等待 | 记录容量收益转移到哪里 | 只比较峰值而忽略 step time、吞吐与质量 |

![训练状态生命周期与显存峰值重叠](../../public/topic_discussion/performance_optimization/training_state_peak_overlap.svg)

## 后训练 workload 如何改变资源账本

后训练并不是统一的一种运行方式。SFT 仍接近单模型训练；QLoRA 改变可训练参数和基座表示；DPO 引入 policy/reference 与成对响应；PPO、GRPO 和 RLVR 则把采样、reward/verifier 与更新拆成不同阶段。账本必须先区分训练形态，再比较策略。

| 训练形态 | 新增或变化的资源对象 | 主要性能问题 | 最小证据 |
|:---|:---|:---|:---|
| SFT / 全参训练 | 全量参数、梯度、optimizer state、activation | 更新状态和 activation 是否超过预算 | step time、tokens/s、peak memory、验证质量 |
| LoRA / QLoRA | adapter、冻结或量化基座、较小 optimizer state | 容量下降是否换来 kernel、反量化或数据路径开销 | 可训练参数、峰值、吞吐、adapter artifact、质量 |
| DPO | policy、reference、chosen/rejected 两路响应 | 多次 forward 与响应长度分布是否主导 step | policy/reference 驻留、pair tokens、step time、偏好质量 |
| PPO | actor、reference、reward、critic、rollout buffer | 多模型驻留和 rollout/update 比例是否失衡 | rollout time、update time、峰值、reward 与任务质量 |
| GRPO / RLVR | policy、reference、每个 prompt 的多响应、reward/verifier 结果 | 组采样和 verifier 是否限制有效更新吞吐 | responses/prompt、rollout throughput、verifier time、有效样本率 |

![后训练 workload 的资源路径](../../public/topic_discussion/performance_optimization/post_training_resource_paths.svg)

## 判断框架：先找峰值对象，再比较代价

| 现象 | 先确认的账本证据 | 可比较的候选 | 不能省略的复测 |
|:---|:---|:---|:---|
| activation 主导峰值 | 层级保存量、序列长度、forward/backward 重叠 | checkpoint、offload、缩短长度、累积 | step time、吞吐、质量与失败状态 |
| optimizer state 主导峰值 | 参数规模、optimizer、参数副本 | 低精度状态、ZeRO、替代 optimizer | 通信、收敛和恢复路径 |
| 通信缓冲区或临时张量抬峰值 | collective 时间线、bucket、workspace 生命周期 | bucket / 并行配置、算子路径 | 通信暴露、扩展效率与正确性 |
| rollout 或 verifier 主导总时长 | rollout/update 比例、多响应数量、环境或 verifier 等待 | 并行采样、批次、backend、异步验证 | 策略新鲜度、有效样本率、质量与失败重试 |
| 多模型驻留超过预算 | policy/reference/reward/critic 的设备位置与精度 | 分池、offload、共享基座、阶段化执行 | handoff、通信、吞吐和数值一致性 |

策略不应按“最省显存”排序，而应按目标预算下是否保留质量、是否满足 step time/SLO、是否引入可接受的通信和维护代价排序。

## 连接到 Part00–02

| 作用 | 入口 | 本页如何使用 |
|:---|:---|:---|
| 输入供给与搬运 | [Part02 · PERF-INPUT 输入流水线](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.md) | 区分取批等待、无效 padding、H2D 与模型状态峰值，不把输入饥饿误判为计算瓶颈 |
| 内存对象与训练机制 | [Part00 · 18 Memory Profiling](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)、[Part02 · 43 统一训练内存](../../02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.md) | 建立统一账本字段 |
| activation 策略 | [Part02 · 19 Checkpoint](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.md)、[42 Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.md) | 比较重算与搬运如何交换时间和容量 |
| activation 与更新节奏前置 | [09 激活与损失反向](../../02_PyTorch_Algorithms/09_Activation_and_Loss_Backward.md)、[10 Attention 反向](../../02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.md)、[11 梯度累积](../../02_PyTorch_Algorithms/11_Gradient_Accumulation.md) | 区分单次 forward 峰值、有效 batch 与 optimizer update 节奏 |
| 多卡状态分摊 | [Part01 · 06 VRAM 与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)、[Part02 · 79 并行基准](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) | 记录状态切分后的通信与峰值变化 |
| 项目证据 | [Part02 · 75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md)、[76 Checkpoint / Offload 对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) | 固定预算与质量门槛后选择策略 |
| SFT 与低资源微调 | [Part02 · 62 SFT](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)、[63 LoRA](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.md)、[65 QLoRA](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) | 比较全参、adapter 和量化基座的状态、吞吐与交付物 |
| 偏好与 RL 训练 | [Part02 · 84 DPO](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md)、[85 GRPO](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.md)、[86 在线 DPO](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.md) | 区分多模型驻留、rollout、verifier、更新与在线回归成本 |

## 项目出口

完成本页后，应得到一份训练资源账本与策略对照：baseline/candidate 的驻留对象、峰值来源、阶段时间、吞吐、质量门槛、artifact、通信影响和失败记录。涉及在线采样时，还要记录 rollout/update 比例、verifier 或环境等待和有效样本率；下一步进入 Task4 判断哪些交接仍位于关键路径。
