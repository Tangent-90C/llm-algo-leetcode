# 性能优化专题术语表

本表解释从瓶颈证据走到候选决策时容易混淆的概念。指标、执行轨迹、性能剖析和瓶颈归因的定义见[性能分析专题术语表](../performance_analysis/glossary.md)；全教程写作遵循[术语表与中英文写作规范](../../docs/terminology.md)。

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 目标指标 | target metric | 本轮优化直接希望改善且必须写明单位、聚合口径和工作负载的指标 | 同时观察到的所有指标 | [01 性能问题与比较契约](./01_performance_problem_and_benchmark.md) |
| 守护条件 | guardrail | 候选必须同时满足的正确性、质量、失败率、容量或成本边界 | 目标指标本身 | [01 性能问题与比较契约](./01_performance_problem_and_benchmark.md) |
| 比较基线 | baseline | 在相同工作负载、环境和测量协议下作为参照的执行方案 | 任意一次历史运行 | [01 性能问题与比较契约](./01_performance_problem_and_benchmark.md) |
| 候选机制 | candidate mechanism | 针对已知约束提出、可通过配对实验被证伪的改动 | 尚未说明作用对象的优化建议 | [02 资源约束与候选代价](./02_observability_and_profiling.md) |
| 主约束 | primary constraint | 当前最直接限制目标指标的资源上界、容量边界或关键路径依赖 | 累计耗时最大的算子 | [02 资源约束与候选代价](./02_observability_and_profiling.md) |
| 候选契约 | candidate contract | 把主约束、作用对象、预期减少量、新增代价、守护条件和证伪条件绑定成一个可测试候选 | 只有名称和开关的优化建议 | [PERF-CANDIDATE](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) |
| 候选代价 | candidate cost | 候选新增的计算、搬运、同步、内存、质量或运维负担 | 只记录候选收益 | [PERF-CANDIDATE](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) |
| 代价转移 | cost transfer | 候选减少一种开销，却把时间或容量压力转移到另一阶段或资源 | 主约束已经消失 | [PERF-CANDIDATE](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) |
| 暴露时间 | exposed time | 未被有效工作隐藏、实际延长关键路径的计算、通信、搬运或等待时间 | 某事件的原始持续时间 | [PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) |
| 执行路径证据 | execution-path evidence | 证明候选实际命中目标加载器、计算内核、数据类型、缓存或调度路径的记录 | 配置中声明启用了候选 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 单候选准入 | single-candidate gate | 在组合策略前，独立检查一个候选的正确性、质量、性能、容量和成本 | 看到局部速度提升即接受 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 策略交互 | strategy interaction | 多个候选同时启用后出现的收益叠加、抵消、资源争用或新依赖 | 单项结果的简单相加 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 边际收益 | marginal gain | 在已经启用其他候选后，再加入一个候选带来的实测增量 | 两个单项收益的算术和 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 瓶颈转移 | bottleneck shift | 原主约束减弱后，另一资源、阶段或依赖成为新的主要限制 | 优化失败或性能回归 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 容量门槛 | capacity gate | 显存、内存、缓存、队列或并发上限必须满足的硬约束 | 平均利用率目标 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 回归阈值 | regression threshold | 结合噪声分布和业务容忍度预先设定的退化边界 | 单次样本差值 | [PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) |
| 回退目标 | rollback target | 组合失败后明确恢复到基线、某个稳定单项候选或上一已接受组合 | 只有失败条件而没有恢复对象 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 回退条件 | rollback condition | 候选触发质量、正确性、容量、成本或稳定性失败后恢复到既定目标的条件 | 事后人工判断 | [07 策略组合与瓶颈转移](./07_end_to_end_performance_engineering.md) |
| 条件化结论 | conditional conclusion | 只对已固定的模型、输入、硬件、后端和测量协议成立的结论 | 对所有环境都成立的普遍结论 | [实验资产表](./experiment_assets.md) |

## 使用规则

- 正文首次出现术语时使用“中文主称（English）”，后续使用中文主称或表中列出的通用缩写。
- `CPU`、`GPU`、`CUDA`、`I/O`、`SLO`、`TTFT`、`TPOT`、`P50`、`P95`、`P99` 可直接使用；普通叙述不直接使用 workload、baseline、candidate、guardrail、artifact、backend、kernel。
- 工具名、API、配置键、命令和结果 JSON 字段保留原始英文，不为中文化修改可执行标识符。
- 性能结论必须同时说明工作负载、目标指标、守护条件和证据范围；不使用“性能更好”“资源更省”作为独立结论。
- 术语修订逐处判断，不对 Notebook、SVG、代码注释或结果 JSON 做无上下文批量替换。
