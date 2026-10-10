# 02 可重复 Baseline 与测量契约

Baseline 的作用不是提供一个“看起来合理”的数字，而是让后续 trace、profile 和 candidate 都能回到同一组运行条件。模型、workload、环境或统计方法发生变化时，应建立新的 baseline，不能继续与旧结果直接比较。

## 固定环境、工作负载与测量过程

| 条件组 | 必须记录 | 常见错误 |
|:---|:---|:---|
| 环境 | 硬件、驱动、CUDA、框架、backend 与版本 | 只写 GPU 型号，不记录软件执行路径 |
| workload | 模型 revision、输入/输出长度、batch/并发、数据选择与随机种子 | baseline/candidate 使用不同输入或请求分布 |
| 测量 | warmup、repeat、同步点、统计量和超时 | 混用首次运行、稳态结果和主机异步计时 |
| 守护条件 | 正确性、质量、OOM、失败率和 SLO | 只保留最快一次，忽略失败与质量回归 |

先完成环境和输入预检，再执行 warmup；正式重复运行必须记录每次结果，不能只保存聚合均值。失败运行保留失败阶段和原因，不从分布中静默删除。

## Baseline 如何交给后续任务

Task2 的 trace 与 Task3 的 profile 必须引用同一 workload 身份和环境快照；Task5 的回归判断再使用相同 baseline/candidate 口径检查分布变化。这样可以区分“执行路径真的改变”与“运行条件漂移”。

| 后续任务 | 从 baseline 继承什么 | 新增什么 |
|:---|:---|:---|
| Trace | workload、环境、阶段定义 | 事件时间线、同步与关键路径 |
| Profile | workload、环境、warmup | 算子、状态、缓冲区和驻留证据 |
| Regression | baseline 分布、失败样本、质量门槛 | candidate 分布、阈值和回退判断 |

![可重复 Baseline 的冻结与继承](../../public/topic_discussion/performance_analysis/reproducible_baseline_contract.svg)

## 项目出口与相关阅读

[Part00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.md)提供基础指标与实验字段；真实训练、推理和多卡项目分别使用自己的 workload，但复用本页的 baseline 契约。

相关阅读：[PyTorch Benchmark Utilities](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)。
