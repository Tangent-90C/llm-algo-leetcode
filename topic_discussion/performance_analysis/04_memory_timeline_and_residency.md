# 04 显存时间线与驻留状态

显存峰值不只决定“是否会 OOM”；对象驻留过久、频繁分配或分配器保留也可能拉长关键路径。这里先读取状态证据，再决定进入哪一种资源策略。

## 先定位峰值与生命周期

把峰值放回运行阶段，区分活跃张量、分配器保留、临时工作区与长生命周期状态。`allocated` 与 `reserved` 的差异是线索，不是对碎片或策略的自动诊断。

| 观察 | 可以提出的假设 | 仍需补充的证据 |
|:---|:---|:---|
| `allocated` 峰值抬高 | 活跃对象容量增加 | 对象类型、shape 与生命周期 |
| `reserved` 明显更高 | 保留或碎片可能存在 | snapshot、分配频率与复用情况 |
| 激活跨反向传播驻留 | 训练状态压力上升 | 计算图与激活检查点对照 |
| 缓存随并发增长 | 请求状态容量受限 | 上下文、并发与淘汰策略 |

![显存时间线与驻留诊断](../../docs/public/topic_discussion/performance_analysis/memory_timeline_diagnosis.svg)

[Part 02 · PERF-MEMORY-TIMELINE 显存驻留与峰值重叠](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)把分配与释放事件重建为活动对象集合，并用真实 GPU 对照检查生命周期变化是否降低峰值。

## 将证据交给策略专题

训练状态的账本与激活检查点 / 卸载策略见[反向传播与训练机制](../backpropagation_training_mechanism/intro.md)和[性能优化](../performance_optimization/intro.md)；KV Cache 的表示与复用见[推理优化](../inference_optimization/intro.md)。选择策略前，先确认对象身份、驻留区间、峰值重叠和回收时刻。

## 进入容量策略复测

可用 [Part 00 · 18 显存分析](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb)练习时间线读取，再进入[性能优化 · 训练输入与状态](../performance_optimization/03_training_memory_ledger.md)或[性能优化 · 推理状态容量](../performance_optimization/04_inference_memory_and_kv_cache.md)，比较容量收益与时间、质量代价。
