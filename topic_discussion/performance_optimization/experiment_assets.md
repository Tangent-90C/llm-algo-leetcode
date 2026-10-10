# 性能优化实验资产表

本表登记性能优化实验复用的环境、输入证据、验证顺序和结果路径。环境创建复用[性能分析实验 Quickstart](../performance_analysis/quickstart.md)，精确依赖与硬件快照见[GPU 环境资产表](../../docs/gpu_environment_assets.md#perf-cuda-mechanism)。本页不复制安装命令、逐包版本或性能分析工具说明。

## 环境与证据分工

| 环境或证据层 | 直接入口 | 可以回答的问题 | 不能替代 |
|:---|:---|:---|:---|
| CPU 机制练习 | [Part 00 · 17 性能分析基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)至[Part 00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb)、Part 01 资源账本、机制页 Step1–4 | 公式、状态关系、硬约束、候选逻辑和测试不变量 | 真实 GPU 执行路径与端到端收益 |
| `PERF-CUDA-MECHANISM` | `PERF-INPUT`、`PERF-IO`、`PERF-CANDIDATE`、`PERF-COMPOSITION`；复用 `PERF-TRACE`、`PERF-MEMORY-TIMELINE`、`PERF-ROOFLINE`、`PERF-REGRESSION` | 单卡输入流水线、对象驻留、资源上界、关键路径、I/O 背压、单候选与组合复测 | 生产服务、多卡通信与远端存储结论 |
| 上游专题实测 | 训练、推理、量化、通信与算子机制结果 | 候选的正确性、质量、容量和执行路径 | 本专题的同口径配对复测与组合决策 |
| 真实系统环境 | 服务、多卡或生产等价环境 | SLO、扩展效率、故障与成本 | 局部机制页对根因的解释 |

## 验证顺序

| 顺序 | 验证对象 | 本阶段回答的问题 | 进入下一步的条件 |
|:---|:---|:---|:---|
| 0 | 环境与有效运行 | 环境、仓库运行时、设备和结果目录是否一致，运行是否无异常 | 环境预检通过；失败运行单独保存 |
| 1 | 比较契约 | 工作负载、目标指标、守护条件、基线和停止条件是否固定 | 基线与候选只有一个预期差异 |
| 2 | 约束证据 | 资源上界、关键路径或对象驻留是否支持当前主约束 | 至少一种判别实验能够证伪归因 |
| 3 | 单项候选 | 候选是否命中目标执行路径并改善目标指标 | 正确性、质量、容量和回归门槛同时通过 |
| 4 | 组合复测 | 多候选同时启用后收益是否保留，瓶颈是否转移 | 保存基线、单项和组合的配对样本与新归因 |
| 5 | 决策交付 | 当前证据支持接受、继续调优还是回退 | 环境、工作负载、失败、成本和回退条件完整 |

## 机制实验注册表

| 实验 ID / 页面 | 固定对照 | 结果资产 | 当前状态 |
|:---|:---|:---|:---|
| `PERF-INPUT` / [输入流水线](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.ipynb) | 同一批数据与设备计算：普通内存 → 锁页内存 → 双 worker 预取 | `benchmarks/results/perf_input_pipeline/<semantic_id>_<run_label>.json` | RTX 5070 Ti 已验证：锁页路径中位数最低；预取路径等待、P95 和吞吐最佳 |
| `PERF-IO` / [I/O 关键路径](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb) | 64 MiB 本地产物与 8 批流水线：串行复制+计算 → 双缓冲重叠 | `benchmarks/results/perf_io_critical_path/<semantic_id>_<run_label>.json` | RTX 5070 Ti 已验证：中位数 7.07 → 4.96 ms，1.42×；产物与流水线正确性通过 |
| `PERF-TRACE` / [关键路径](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) | 同一事件图：原始持续时间 → 依赖、重叠与暴露等待 | `benchmarks/results/perf_trace/<workload_id>_<run_label>.json` | 性能分析主归属；本专题只消费稳定结果 |
| `PERF-MEMORY-TIMELINE` / [显存驻留](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb) | 同一输出：状态重叠 → 释放与重算候选 | `benchmarks/results/perf_memory_timeline/<workload_id>_<run_label>.json` | 性能分析主归属；本专题消费峰值阶段与对象身份 |
| `PERF-ROOFLINE` / [资源上界](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) | 实测带宽、实测计算上界与候选工作量账本 | `benchmarks/results/perf_roofline/<workload_id>_<run_label>.json` | 性能分析主归属；本专题消费约束证据 |
| `PERF-REGRESSION` / [配对回归](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) | 对齐顺序的基线/候选配对样本 | `benchmarks/results/perf_regression/<workload_id>_<run_label>.json` | 性能分析主归属；本专题消费回归与准入结论 |
| `PERF-CANDIDATE` / [约束到候选](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) | 同一 SwiGLU 流水线：FP32 串行 → BF16 串行 / FP32 双缓冲 | `benchmarks/results/perf_candidate/<semantic_id>_<run_label>.json` | RTX 5070 Ti v2 已验证：BF16 两档约 2.65×/2.88×且活动字节减少 50%；FP32 重叠仅在 256 Token 达到采用门槛 |
| `PERF-COMPOSITION` / [策略组合](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) | FP32 串行、BF16 串行、FP32 重叠与 BF16+重叠同次复测 | `benchmarks/results/perf_composition/<semantic_id>_<run_label>.json` | RTX 5070 Ti v2 已验证：组合相对 BF16 单项约 +5.6%/+12.2%，两档均为 24/24 轮正边际且 P95 未回归 |

## 上游证据消费契约

| 输入证据 | 必须携带 | 本专题如何使用 |
|:---|:---|:---|
| 训练机制结果 | 模型、批次、序列长度、状态生命周期、正确性和 step 时间 | 比较重算、卸载、输入流水线和状态驻留候选 |
| 推理机制结果 | 模型、请求分布、上下文、生成长度、缓存、批次和延迟分位数 | 比较缓存、批处理、调度与执行路径候选 |
| 量化与压缩结果 | 模型 commit、量化配置、质量、产物和实际执行路径 | 判断容量收益是否换来质量或计算回退 |
| 通信与并行结果 | world size、拓扑、载荷、最慢进程和暴露通信 | 判断切分、重叠与负载均衡是否改变关键路径 |
| 算子结果 | 输入形状、数据类型、计算内核、正确性、预热和重复样本 | 判断局部收益能否在端到端工作负载中保留 |

性能优化结论不直接读取另一 Notebook 的内部变量，只消费结果 JSON、执行轨迹或产物清单。输入、硬件、后端、候选实现或测量协议改变时使用新的运行标签，不覆盖历史结果。
