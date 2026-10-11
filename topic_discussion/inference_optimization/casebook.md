# 推理优化判断手册

使用本表前，固定模型与分词器版本、部署产物、执行后端、数据类型、请求分布、输入/输出长度、并发度和预热规则。至少保留 TTFT、TPOT、吞吐、P99、峰值显存、质量结果、真实执行路径和失败记录。

从“现象或目标”列定位请求阶段和状态对象，再采集最小证据。还不能区分预填充、解码、缓存、排队或加载路径时，不直接更换推理引擎或增加批量。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 长输入下首词元明显变慢 | 预填充计算、访存、分块和重复前缀 | TTFT、预填充时间、输入长度、峰值显存和计算内核路径 | [02 预填充与注意力计算内核](./02_prefill_and_attention_kernel.md) | [Part 02 · 20 FlashAttention / SDPA](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) / [算子优化](../operator_optimization/intro.md) |
| 单请求生成阶段每词元很慢 | 解码计算、KV 读取、候选生成与验证 | TPOT、平均接受长度、验证成本、质量与回退记录 | [03 解码策略](./03_decoding_strategies.md) | [Part 02 · 68 投机解码基准](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb) |
| 并发增加后 TPOT 或 P99 上升 | 队列等待、单轮预算、动态批次与阶段混合 | 请求时间线、批次组成、每轮词元数、TPOT、P99 和吞吐 | [07 Serving 调度、PD 与异构路由](./07_serving_scheduling_and_pd.md) | [Part 02 · 70 Serving 调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) |
| KV Cache 持续增长，上下文或并发无法再增加 | 每词元状态、块利用率、驱逐、抢占与回收 | 容量账本、块表、引用计数、驱逐/抢占事件和 OOM 阶段 | [04 KV Cache 表示、管理与复用](./04_kv_cache_lifecycle_and_reuse.md) | [Part 02 · 37 KV 生命周期](../../02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb) |
| 重复前缀很多，但 TTFT 没有下降 | 连续命中长度、状态隔离键和真实复用词元数 | 请求前缀、命中计数、未命中后缀、后端日志和 TTFT 对照 | [04 KV Cache 表示、管理与复用](./04_kv_cache_lifecycle_and_reuse.md) | [Part 02 · 69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) |
| 模型或量化文件可以加载，但显存或速度收益不稳定 | 产物格式、加载器、数据类型、计算内核与静默回退 | 产物清单、加载日志、实际计算内核、峰值显存、延迟和质量对照 | [05 量化推理与部署](./05_quantized_inference_and_deployment.md) | [Part 02 · 82 量化产物准入](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb) / [Part 02 · 83 Serving 复测](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.ipynb) |
| 显存降低，但交互延迟或质量变差 | 权重、激活或 KV 量化对象，以及调度和回退路径 | TTFT、TPOT、P99、峰值显存、任务质量和真实数据类型 | [05 量化推理与部署](./05_quantized_inference_and_deployment.md) | [量化与压缩](../quantization_compression/intro.md) / [06 端到端基准与决策](./06_benchmark_and_decision.md) |
| 单卡容量或吞吐不足 | 请求是否可独立切分，模型是否必须跨卡，通信是否暴露 | 单卡基线、负载分布、跨卡字节、扩展效率和恢复记录 | [08 分布式推理与并行扩展](./08_distributed_inference_and_parallelism.md) | [通信与并行](../communication_parallelism/intro.md) / [Part 02 · 81 分布式推理](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb) |
| 同一结果在重复运行中方向不稳定 | 请求顺序、预热、缓存状态、采样参数和后台负载 | 原始重复样本、正反顺序对照、失败运行和环境快照 | [06 端到端基准与决策](./06_benchmark_and_decision.md) | [性能分析](../performance_analysis/intro.md) / [性能优化](../performance_optimization/intro.md) |
