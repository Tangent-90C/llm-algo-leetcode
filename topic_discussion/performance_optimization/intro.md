# 性能优化（Performance Optimization）

> 专题类型：主学习路线　能力出口：把瓶颈证据转化为候选策略、配对复测与工程决策

性能优化从可复查的基线和瓶颈归因出发，选择能够改变主约束的候选，并检查代价是否转移到质量、容量、延迟、通信或成本。学习主线是“固定目标 → 读取归因 → 选择候选 → 配对复测 → 检查瓶颈转移 → 准入与组合决策”。

[性能分析](../performance_analysis/intro.md)回答“时间和资源去了哪里”；本专题回答“应比较什么、改动后代价转移到哪里、候选是否值得保留”。训练、推理、量化、通信和计算内核的具体改动分别进入对应专题验证。

## 学习路线

按 Task 顺序学习时使用下表；需要沿一个性能问题连续走到决策时进入[性能优化问题链](./walkthrough.md)，已有瓶颈或候选时使用[性能优化判断手册](./casebook.md)。

![性能优化学习路线](../../docs/public/topic_discussion/performance_optimization/performance_optimization_roadmap.svg)

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 工作负载、指标与比较基线 | 当前固定工作负载要改善哪个指标，哪些正确性、质量、容量和成本条件不能被破坏？ | 核心：[Part 00 · 19 异常定位](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb) → [Part 00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb)；扩展：[Part 00 · 17 性能分析基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb) | 确认运行有效 → 固定工作负载 → 选择目标指标 → 设置守护条件 → 建立基线与停止条件 | [专题正文 · 01 性能问题与比较契约](./01_performance_problem_and_benchmark.md) |
| Task1 · 资源约束与候选代价 | 当前受计算、带宽还是容量约束，候选会减少哪类工作量并新增什么代价？ | 核心：[Part 01 · 02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb) → [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) → [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb) → [Part 02 · PERF-ROOFLINE 资源上界](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) → [Part 02 · PERF-CANDIDATE 约束到候选](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb)；扩展：[性能分析 · Task4 瓶颈归因](../performance_analysis/intro.md) | 工作量账本 → 资源上界 → 候选作用对象 → 预期减少量 → 新增代价与证伪条件 | [专题正文 · 02 资源约束与候选代价](./02_observability_and_profiling.md) |
| Task2 · 训练输入与状态优化 | 输入、激活、梯度、优化器状态和通信缓冲区怎样共同限制训练时间与峰值？ | 核心：[Part 00 · 18 显存分析](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb) → [Part 01 · 06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) → [Part 01 · 07 CPU/GPU 异构调度](../../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb) → [Part 02 · PERF-INPUT 输入流水线](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.ipynb)与[PERF-MEMORY-TIMELINE 显存驻留](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)；扩展：[Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.ipynb)、[Part 02 · 42 激活卸载](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb)、[Part 02 · 43 训练内存账本](../../02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.ipynb) | 输入供给 → 训练步骤时间线 → 状态驻留 → 候选对照 → 时间—容量取舍 | [专题正文 · 03 训练输入、状态与资源账本](./03_training_memory_ledger.md) |
| Task3 · 推理请求与状态优化 | 请求分布、KV Cache、批量和并发怎样共同限制延迟、吞吐与容量？ | 核心：[Part 01 · 11 KV Cache 与显存增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) → [Part 02 · INF-ITERATION-BUDGET 迭代成本与批量上界](../../02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.ipynb) → [Part 02 · 22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [Part 02 · 37 KV Cache 生命周期](../../02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb) → [Part 02 · INF-CONTINUOUS-BATCHING 连续批处理](../../02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching.ipynb)；扩展：[推理优化](../inference_optimization/intro.md) | 固定请求分布 → 状态容量 → 批量与并发 → 延迟与吞吐 → 候选对照 | [专题正文 · 04 推理状态容量与 KV Cache](./04_inference_memory_and_kv_cache.md) |
| Task4 · 计算、通信与 I/O 关键路径 | 哪些计算、搬运、集合通信或队列等待真正暴露在关键路径上，依赖关系允许隐藏多少？ | 核心：[Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb) → [Part 01 · 17 CUDA 流与异步执行](../../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb) → [Part 02 · PERF-TRACE 关键路径](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) → [Part 02 · PERF-IO I/O 关键路径](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb)；扩展：[Part 01 · 26 并行策略决策](../../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.ipynb)、[Part 02 · 47 NCCL 性能分析](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb) → [Part 02 · 48 通信热点与缓解](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.ipynb) | 重建依赖 → 区分总时间与暴露时间 → 定位关键路径 → 调整重叠或同步 → 复测 | [专题正文 · 05 计算、通信与 I/O](./05_compute_communication_and_io.md) |
| Task5 · 单候选执行与准入 | 候选是否真正命中目标执行路径，并同时通过正确性、质量、性能回归和成本门槛？ | 核心：[Part 02 · PERF-REGRESSION 配对基准与回归阈值](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) → [Part 01 · 33 TCO 与成本模型](../../01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb) → [Part 01 · 34 异构加速器与运行时约束](../../01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb)；扩展：各分支机制页的真实执行结果 | 正确性与质量 → 执行路径 → 配对性能分布 → 资源成本 → 单候选决策 | [专题正文 · 06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| Task6 · 策略组合与瓶颈转移 | 单项候选组合后收益是否保留，主约束转移到了哪里，何时必须回退？ | 核心：[Part 02 · PERF-COMPOSITION 策略相互作用与瓶颈转移](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) | 读取候选契约、分支结果与准入结论 → 固定同一基线 → 复测单项候选 → 组合候选 → 重新归因 → 检查回归与回退条件 → `accept / tune / reject` | [专题正文 · 07 策略组合与瓶颈转移](./07_end_to_end_performance_engineering.md) |

下图把训练状态、推理状态、计算、通信、I/O 和最终准入放在同一张知识地图中。训练、推理、量化、通信和算子专题提供候选机制；本专题固定比较条件，判断候选是否改变主约束，并在组合后重新归因。

![性能优化知识地图](../../docs/public/topic_discussion/performance_optimization/performance_optimization_knowledge_map.svg)

## 扩展与跨专题入口

### 按需回补

| 需要补充的机制 | 回补入口 | 回到性能优化时需要带回的证据 |
|:---|:---|:---|
| 低精度表示与计算路径 | [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) → [Part 01 · 12 Tensor Core 与混合精度](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb) → [Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) | 表示字节、累加精度、计算内核与回退路径 |
| Attention 访存与推理工作量 | [Part 01 · 04 Attention 显存优化](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb) → [Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb) | Attention 中间状态、访存量和融合收益条件 |
| 计算内核、融合与片上存储 | [Part 01 · 08 CUDA/Triton 编程模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb) → [Part 01 · 16 Warp、Block 与 Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb) → [Part 01 · 19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb) → [Part 01 · 23 Tensor Core 深入](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.ipynb) → [Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb) | 计算内核正确性、启动次数、访存量、占用率与片上复用 |
| 局部计算或访存瓶颈已经确认 | [算子优化](../operator_optimization/intro.md) | 计算内核语义、图改写、微基准与模型工作负载复测 |
| 动态形状与编译图 | [Part 01 · 09 AI 编译器与计算图优化](../../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb) → [Part 01 · 30 动态 Shape](../../01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb) | 图捕获范围、重编译次数、形状分布和稳态收益 |
| 容错、检查点与恢复成本 | [Part 01 · 28 容错与检查点](../../01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.ipynb) | 保存频率、恢复时间、写回背压与失败成本 |
| activation、梯度或优化器状态占用 | [反向传播与训练机制](../backpropagation_training_mechanism/intro.md) | 状态生命周期、正确性、峰值与 step 时间 |
| 请求、量化、并行与输入数据 | [推理优化](../inference_optimization/intro.md)、[量化与压缩](../quantization_compression/intro.md)、[通信与并行](../communication_parallelism/intro.md)、[数据工程](../data_engineering/intro.md) | 请求分布、质量与容量、通信暴露、样本长度和输入吞吐 |

旧显存优化内容已经按对象拆入训练状态、推理状态、量化和通信入口；历史链接的对应关系见[旧显存路线迁移说明](../memory_performance_tuning/intro.md)，不再沿旧目录继续学习。

## 实验与项目出口

性能优化使用同一条证据升级链：先取得性能分析证据，再验证局部机制和执行路径，最后进入真实工作负载的单项与组合复测。

| 证据阶段 | 要回答的问题 | 主要出口 |
|:---|:---|:---|
| 归因输入 | 主瓶颈和回归阈值是否已有可复查证据？ | Task0 的有效运行与比较基线；已有完整诊断时复用性能分析记录 |
| 局部候选 | 候选是否改变目标对象，正确性与质量是否保持？ | 各机制 Notebook 的 CPU / GPU 结果 |
| 单项准入 | 产物、执行后端、计算内核、SLO 和成本是否同时通过？ | 分支机制结果与单候选准入记录 |
| 组合复测 | 多项候选组合后是否出现收益抵消、瓶颈转移或新回归？ | 固定工作负载下的基线、单项候选与组合候选配对结果 |
| 工程决策 | 当前证据支持接受、继续调优还是回退？ | 含环境、工作负载、质量、资源、失败与回退条件的结论 |

CPU 适合验证公式、账本、状态关系和决策逻辑；GPU 或真实执行后端用于确认峰值显存、延迟、吞吐、通信暴露和执行路径。实验只在结果中记录本次版本和硬件快照，统一环境与环境配置（Profile）查询[GPU 环境资产表](../../docs/gpu_environment_assets.md)。

首次运行从[实验资产表](./experiment_assets.md)确认实验身份、验证顺序和结果路径；术语差异查询[专题术语表](./glossary.md)。
