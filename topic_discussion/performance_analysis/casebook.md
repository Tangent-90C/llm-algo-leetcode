# 性能分析判断手册 | Profiling Casebook

使用本页时，先确认 workload 与 baseline 是否完整；再根据观察到的事实选择测量，而不是直接跳到某个优化策略。

## 症状到测量的快速映射

| 观察到的现象 | 先记录的指标 | 优先证据 | 可能转入的专题 |
|:---|:---|:---|:---|
| step 或请求延迟上升 | 阶段时间、P50 / P99、TTFT / TPOT | `torch.profiler`、Chrome Trace | 性能优化 Task4、推理优化 |
| GPU 出现空洞 | CPU/GPU 时间线、同步与 H2D | profiler / Nsight Systems | 性能优化 Task4 |
| 峰值显存抬高或 OOM | allocated、reserved、峰值阶段 | memory timeline / snapshot | 性能优化 Task2 / Task3 |
| 多卡没有线性扩展 | 每卡阶段时间、collective、通信暴露 | NCCL trace、Nsight Systems | 通信与并行 |
| 候选看起来更快 | 均值、分位数、失败率、质量 | baseline / candidate 重复对照 | 性能优化 Task5–6 |

## 证据强度与行动

| 当前证据 | 可得结论 | 下一步 |
|:---|:---|:---|
| 单次计时、单张截图 | 有现象，不足以归因 | 补 workload、warmup 与 baseline |
| trace 指向某阶段 | 有待验证的瓶颈假设 | 检查输入、状态、通信或 shape |
| 同口径候选对照 | 候选影响了目标指标 | 检查质量、波动、尾延迟与失败 |
| 重复对照且守护指标通过 | 可形成带条件的工程结论 | 记录适用条件并进入组合复测 |

## 最小记录字段

`现象 → 指标 → workload → 环境 → baseline / candidate → 证据位置 → 质量与回归 → failure → action`

训练项目可参照 [73–76](../../02_PyTorch_Algorithms/2_10.md)，推理与 serving 可参照 [66 / 67 / 70](../../02_PyTorch_Algorithms/2_10.md)，多卡项目可参照 [79–81](../../02_PyTorch_Algorithms/2_10.md)。
