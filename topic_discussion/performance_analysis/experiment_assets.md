# 性能分析实验资产表

本表登记性能分析实验使用的环境、工作负载、结果路径和证据状态。环境创建见[实验 Quickstart](./quickstart.md)，精确依赖与硬件快照见[GPU 环境资产表](../../docs/gpu_environment_assets.md#perf-cuda-mechanism)。本页不复制安装命令或逐包版本。

## 环境与工具角色

| 环境或工具层 | 直接入口 | 可以形成的证据 | 不能替代 |
|:---|:---|:---|:---|
| CPU 基础练习 | [Part 00 · 17 性能分析基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)至[Part 00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb)、Part 01 资源与执行基础、机制页 Step1–4 | 有效性、指标、账本、事件依赖、资源上界、归因和回归规则 | 真实 GPU 时间线与资源利用率 |
| `PERF-CUDA-MECHANISM` | 本专题：`PERF-TRACE`、`PERF-MEMORY-TIMELINE`、`PERF-ROOFLINE`、`PERF-REGRESSION`；共享环境中的性能优化页面：`PERF-INPUT`、`PERF-IO` | 单卡执行轨迹、对象驻留、经验资源上界、局部时间和配对分布 | 生产服务、真实存储或多卡扩展结论 |
| Nsight Systems / Compute | 目标机器上的外部采集 | 系统时间线、计算内核资源指标和暴露等待 | 固定工作负载、质量门槛与端到端关键路径 |
| 多卡通信环境 | [Part 02 · 47 NCCL 通信性能分析](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb)与通信专题实验资产 | 集合通信、最慢进程和扩展效率 | 单卡机制结果 |
| 服务环境 | 推理优化专题实验资产 | 请求队列、缓存、TTFT、TPOT 和吞吐 | 训练或计算内核归因 |

## 验证顺序

| 顺序 | 页面或阶段 | 本阶段回答的问题 | 进入下一步的条件 |
|:---|:---|:---|:---|
| 0 | 环境安装与预检 | Python、PyTorch、CUDA、仓库运行时和 Jupyter 内核是否一致 | `pip check` 无冲突；环境预检返回 `status=ok` |
| 1 | [PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) | 实际事件依赖、关键路径和暴露等待能否被重建 | 执行轨迹与摘要 JSON 均生成；事件时间和依赖有效 |
| 2 | [PERF-MEMORY-TIMELINE](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb) | 对象何时驻留、何时形成峰值重叠 | 输出一致；峰值时刻、活动对象和实际峰值字段完整 |
| 3 | [PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) | 计算、访存与端到端等待能否区分 | 工作量账本、经验带宽、经验计算上界和候选效率完整 |
| 4 | [PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) | 候选差异是否超过噪声并满足守护条件 | 配对身份、失败运行、区间和决策字段完整 |
| 5 | 诊断交接 | 多类证据是否支持一个可证伪归因和明确行动 | 记录证据等级、主约束、反例和复查条件 |

## 机制实验注册表

| 实验 ID / 页面 | 固定工作负载 | 主要结果 | 当前状态 |
|:---|:---|:---|:---|
| `PERF-TRACE` / PERF-TRACE | 2048 方阵；逐步同步对比末尾同步；10 次交替顺序配对 | 原始执行轨迹、事件摘要、关键路径和暴露等待 | RTX 5070 Ti 已验证：轨迹完整；25.89 → 25.32 ms，约 2.25% 差异仅作方向性证据 |
| `PERF-MEMORY-TIMELINE` / PERF-MEMORY-TIMELINE | 8192 × 2048 FP32；激活与工作区重叠 → 释放激活后重算 | 输出一致性、allocated / reserved 峰值、延迟分布与峰值决策 | RTX 5070 Ti 已验证：allocated 峰值下降约 33.3%，reserved 不变，延迟约 1.86× |
| `PERF-ROOFLINE` / PERF-ROOFLINE | 256 MiB FP32 设备复制、4096 上界 GEMM 与 2048 候选 GEMM | 经验带宽、经验计算吞吐、算术强度、Roofline 上界与效率 | RTX 5070 Ti 已验证：394.76 GB/s、20.04 TFLOP/s；候选为计算受限并接近经验上界 |
| `PERF-REGRESSION` / PERF-REGRESSION | 1 / 32 / 256 Token、隐藏维度 2048；Eager / Compile 配对顺序和 5% 回归门槛 | 首次调用、逐试验延迟、比例分布、自助区间、失败与逐负载决策 | RTX 5070 Ti 已验证：三档输出一致；1 Token `reject`、32 Token `tune`、256 Token `accept`，总决策 `reject` |
| `PERF-INPUT` / PERF-INPUT | 固定合成样本；普通内存、锁页内存、双 worker 预取三条输入路径 | 输入等待、搬运、步骤中位数/P95、吞吐与正确性 | RTX 5070 Ti 已验证：锁页路径中位数最低；预取路径等待、P95 和吞吐最佳 |
| `PERF-IO` / PERF-IO | 64 MiB 本地产物与 8 批流水线；串行复制+计算对比双缓冲重叠 | 阶段服务时间、完整路径中位数/P95、暴露时间与正确性 | RTX 5070 Ti 已验证：7.07 → 4.96 ms，1.42×；本地重叠候选通过 |

## 工具产物注册表

工具如何选择以及何时升级由[正文 03](./03_time_breakdown_and_trace_reading.md)解释；本表只登记需要保存的产物和字段。

| 工具或证据层 | 结果引用字段 | 最小配套字段 | 主要消费位置 |
|:---|:---|:---|:---|
| CUDA Event / 轻量计时 | `raw_samples_path` | 单位、同步方式、预热、重复次数、失败样本 | Task1、Task5 |
| PyTorch Profiler | `trace_path`、`profiler_summary_path` | schedule、活动类型、shape、调用栈开关 | Task2–3 |
| 显存快照 / 驻留时间线 | `memory_snapshot_path` | 峰值阶段、对象身份、allocated / reserved | Task3 |
| Nsight Systems | `nsys_report_path` | 采集命令、时间窗口、进程、设备 | Task2、Task4 |
| Nsight Compute | `ncu_report_path` | 计算内核、输入形状、指标集合、调用位置 | Task4 |
| NCCL / 多进程追踪 | `nccl_trace_path`、`rank_summary_path` | world size、拓扑、载荷、最慢进程、时钟口径 | Task4 |
| 推理服务追踪 | `service_trace_path` | 请求分布、缓存、批次、调度和后端版本 | 推理优化、性能优化 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`workload_version`、`run_label` |
| 环境 | 环境配置 ID、Python、PyTorch、CUDA、GPU 与外部工具版本 |
| 工作负载 | 输入形状、批次、阶段、预热、重复次数与随机种子 |
| 指标 | 名称、单位、聚合方式、原始样本和失败分母 |
| 路径证据 | 对应产物引用字段、性能剖析摘要、关键路径、对象驻留或计算内核记录 |
| 归因 | 候选瓶颈、判别实验、反例与当前主约束 |
| 回归 | 基线 / 候选配对、区间、守护条件、阈值和失败率 |
| 结论 | `failure`、`evidence_level`、行动和复查条件 |

下游复测只消费稳定结果或产物引用，不复制机制页内部变量。输入、模型、后端、硬件或测量协议改变时使用新的运行标签，不覆盖历史结果。
