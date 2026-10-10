# 04 显存时间线与驻留状态

显存峰值不只决定“是否会 OOM”；对象驻留过久、频繁分配或 allocator 保留也可能拉长关键路径。这里先读取状态证据，再决定进入哪一种资源策略。

## 先定位峰值与生命周期

把峰值放回运行阶段，区分活跃 tensor、allocator 保留、临时 workspace 与长生命周期状态。`allocated` 与 `reserved` 的差异是线索，不是对碎片或策略的自动诊断。

| 观察 | 可以提出的假设 | 仍需补充的证据 |
|:---|:---|:---|
| `allocated` 峰值抬高 | 活跃对象容量增加 | 对象类型、shape 与生命周期 |
| `reserved` 明显更高 | 保留或碎片可能存在 | snapshot、分配频率与复用情况 |
| activation 跨 backward 驻留 | 训练状态压力上升 | 计算图与 checkpoint 对照 |
| cache 随并发增长 | 请求状态容量受限 | 上下文、并发与淘汰策略 |

![显存时间线与驻留诊断](../../public/topic_discussion/performance_analysis/memory_timeline_diagnosis.svg)

## 将证据交给策略专题

训练状态的账本与 checkpoint / offload 策略由[反向传播与训练机制](../backpropagation_training_mechanism/intro.md)和[性能优化 Task2](../performance_optimization/intro.md)解释；KV Cache 的表示与复用由[推理优化](../inference_optimization/intro.md)和性能优化 Task3 解释。本页只保留“何时、何物、驻留多久”的测量证据。

## 项目出口与相关阅读

可用 [Part00 · 18 显存分析](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.md) 练习时间线读取，再通过 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) 与 [76 策略基准](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) 检查策略代价。
