# 07 从归因到行动

性能分析的终点不是一张图，而是一份能够说明“为何采取这个动作、在哪些条件下成立”的记录。行动状态描述分析进度；项目中的 `accept / tune / reject` 则描述具体候选的工程决策，两者可以同时出现。

## 用证据决定行动

| 状态 | 需要具备的证据 | 输出 |
|:---|:---|:---|
| `inspect` | 现象存在，但指标或 workload 还不完整 | 补测字段与待验证假设 |
| `optimize` | 时间、显存或通信证据已指向目标对象 | 候选策略与预期代价 |
| `validate` | 同口径对照显示稳定收益，守护指标未越界 | 适用 workload、复测记录与限制 |
| `revert` | 候选无收益、引入回归或无法复现 | 回退原因与下一条假设 |

![从证据到行动决策](../../public/topic_discussion/performance_analysis/action_decision_board.svg)

## 最小结论记录

每次结论至少保留：现象、指标、workload、环境、baseline / candidate、证据位置、质量或稳定性检查、失败信息与下一步。缺少这些条件时，只能描述一次观察。

## 项目出口与相关阅读

训练项目可沿 [73 → 76 → 75 → 74](../../02_PyTorch_Algorithms/2_10.md) 收集基线、策略、预算和 trace 证据；推理项目可从 [66 / 67 / 70](../../02_PyTorch_Algorithms/2_10.md) 比较 backend、量化与调度；多卡项目使用 [79–81](../../02_PyTorch_Algorithms/2_10.md) 验证扩展与通信。
