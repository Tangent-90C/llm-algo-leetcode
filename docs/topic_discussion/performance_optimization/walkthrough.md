# 性能优化问题链：从现象到工程决策

性能问题通常从一个现象开始：训练 step 变长、显存接近上限、推理 TTFT/P99 变差，或多卡后收益不如预期。下面的路线不预设某种优化技巧，而是先建立比较契约，再根据 workload 分流。

## 第一段：先固定比较条件

先在 [01 性能问题与 Benchmark 基础](./01_performance_problem_and_benchmark.md) 固定模型、版本、dtype、硬件、输入分布、batch/并发、warmup、重复次数、质量门槛与 baseline。只有在这些条件一致时，后续指标才能支持判断。

## 第二段：用证据定位关键路径

先在[性能分析专题](../performance_analysis/intro.md)取得重复计时、profiler、显存统计或 trace 等证据；再进入[02 归因证据与策略假设](./02_observability_and_profiling.md)，将它们整理为“哪个对象或哪段等待值得继续比较”的假设。此时仍不直接选择优化方案。

## 第三段：按 workload 选择资源问题

### 训练：显存账本与 step 时间

当问题发生在 forward、backward、optimizer 或峰值显存时，进入 [03 显存账本与训练侧优化](./03_training_memory_ledger.md)。先识别参数、梯度、optimizer state、activation、workspace 与通信缓冲区的重叠，再比较 checkpoint、offload、累积、分片或低精度策略。

先确认 OOM 出现在加载、forward、backward 还是 update：同样是“显存不够”，对应的驻留对象和可接受的代价并不相同。训练数据读取、CPU 预处理和 H2D 搬运若出现在 step 的关键路径上，先用 [PERF-INPUT](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.md)拆分输入等待，再转入 Task4，而不是误判为 activation 问题。

可用 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) 建立基线，按需要进入 [76 Checkpoint / Offload](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)、[75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) 与 [74 Profiling 驱动优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)。

SFT、QLoRA、DPO 或 GRPO 还要继续区分 adapter、量化基座、policy/reference、reward/verifier 与 rollout 结果。机制与目标进入[后训练优化](../post_training_optimization/intro.md)，资源账本进入[03 训练状态与后训练资源账本](./03_training_memory_ledger.md)；项目证据由 [62](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)、[65](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) 和 [84–86](../../02_PyTorch_Algorithms/2_10.md) 提供。

### 推理：状态容量与请求指标

当问题与长上下文、并发、KV Cache、TTFT 或 TPOT 相关时，进入 [04 推理状态容量与 KV Cache 成本](./04_inference_memory_and_kv_cache.md)。此处先确认每 token 状态成本、请求长度/并发分布、复用命中和质量，再判断分页、缓存、表示压缩或容量预算是否值得继续。

可用 [66 推理性能对比](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) 固定浮点 baseline；复用问题进入 [69 前缀缓存基准](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)，量化与 backend 问题进入 [67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) 与 [83 Serving Profile](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.md)。

### Serving 与多卡：通信、I/O 和重叠

当 trace 显示队列等待、CPU-GPU 搬运、collective 或同步间隙时，进入 [05 计算、通信、I/O 与重叠瓶颈](./05_compute_communication_and_io.md)。先判断该段是否位于关键路径，再验证重叠是否真正缩短 wall time。

I/O 需要按对象拆开记录：训练输入的读取与 H2D、模型或量化 artifact 的加载、KV handoff 等状态传输，以及流式输出、checkpoint 和日志写入。[PERF-IO](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md)负责统一对象、频率、依赖和暴露时间。后训练还需记录 rollout、reward/verifier、权重刷新和 update 的交接。

Serving 调度进入 [70 Serving 调度基准](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)。多卡扩展按对象分流：训练状态进入 [79](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md)，MoE 路由进入 [80](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md)，Serving 进入 [81](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md)。

## 第四段：候选准入与最终复测

在 [06 成本、容量与生产准入](./06_cost_model_and_production_decision.md) 检查候选是否仍满足容量、SLO、质量、稳定性和成本约束。通过准入不等于已经采用；还要在 [07 端到端性能工程与决策](./07_end_to_end_performance_engineering.md) 比较 baseline、单项候选、组合策略与回退方案。

最终报告应写明：收益在哪里出现、代价转移到哪里、结论适用于什么模型/硬件/workload，以及为什么是 `accept`、`tune` 或 `reject`。

![局部候选到端到端性能决策](../../public/topic_discussion/performance_optimization/strategy_combination_decision.svg)
