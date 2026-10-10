# 图级优化与编译支撑：迁移说明

图优化、IR、lowering、legalization、schedule 与 backend 约束已并入[算子优化](../intro.md) **Task1**，不再作为独立专题或并列阅读路线维护。

| 原页面 | 现在的主入口 | 迁移后的职责 |
|:---|:---|:---|
| 01 为什么编译与图优化 | [主专题 01](../01_why_operator_optimization_matters.md) | 图正确与执行高效的区别 |
| 02 图结构与 Fusion 决策 | [主专题 03](../03_fusion_and_kernel_composition.md) | 依赖、中间张量与融合候选 |
| 03 Lowering、Legalization 与 Scheduling | [主专题 03](../03_fusion_and_kernel_composition.md) | 下沉合法性、layout 与调度 |
| 04 执行模型与 Backend 约束 | [主专题 04](../04_cuda_execution_and_hardware_constraints.md) | backend、硬件与 fallback 约束 |
| 05 Backend 成本模型 | [主专题 05](../05_cost_model_and_profiling.md) | backend 分化与成本证据 |
| 06 Benchmark 与项目验证 | [主专题 06](../06_benchmark_and_project_validation.md) | graph / backend / workload 三层验证 |
| walkthrough / casebook | [主专题问题链](../walkthrough.md) / [判断手册](../casebook.md) | 图变换判断流程与失败案例 |

旧页面暂时保留，以兼容历史链接；新增或更新内容只写入主专题页面。
