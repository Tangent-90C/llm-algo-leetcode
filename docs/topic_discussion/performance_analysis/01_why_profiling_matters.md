# 01 测量问题与证据链

训练 step 变慢、TTFT 变高或多卡扩展变差，都只是现象。先把现象写成可测问题，才能让后续 trace 与对照实验真正回答“为什么”。

## 从现象到可测假设

一次有效的性能问题定义同时写清对象、指标、条件与质量约束。这样得到的假设可以被后续证据支持或推翻，而不是把热点名称直接当作结论。

| 要记录的内容 | 示例 | 作用 |
|:---|:---|:---|
| 现象与阶段 | decode 阶段 TPOT 上升 | 限定问题发生的位置 |
| 目标指标 | `TPOT`、`step_time_ms`、峰值显存 | 明确比较方向与单位 |
| workload | 模型、输入长度、batch、并发与采样参数 | 让结果可复现 |
| 假设 | 输入搬运拉长了关键路径 | 提供可证伪的归因方向 |
| 质量门槛 | 输出一致性、任务分数或失败率不能回归 | 防止“更快但不可用” |

![性能分析的证据链](../../public/topic_discussion/performance_analysis/profiling_evidence_chain.svg)

## 证据如何升级

热点、峰值或利用率异常都是观察。只有当它们与固定条件下的 baseline / candidate 对照相互吻合，才可升级为某一类瓶颈的归因；否则应保留为下一轮测量假设。

| 证据状态 | 可以说明什么 | 下一步 |
|:---|:---|:---|
| 单次计时或截图 | 有值得检查的现象 | 固定 workload，建立 baseline |
| trace 与阶段变化一致 | 某阶段可能在拖慢关键路径 | 检查输入、形状、同步或状态变化 |
| 同口径对照可重复 | 候选改动影响了目标指标 | 复核质量、波动与其他指标 |

## 项目出口与相关阅读

基础取证可从 [Part00 · 0E 调试基础](../../00_Prerequisites/0E.md) 与 [Part01 · 13 性能分析](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) 开始；真实项目中的证据收口可参考 [74 Profiling 驱动优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) 与 [79 分布式并行基准](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md)。

相关阅读：[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)。
