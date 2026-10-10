# 03 时间线、I/O 与关键路径

同样是“慢”，可能来自计算量、kernel 启动、输入搬运、同步等待或请求排队。先建立 baseline，再用阶段时间线判断哪一段进入关键路径。

## 建立可比较的时间基线

先固定模型、输入、精度、warmup、重复次数与同步方式；再选择与问题匹配的指标。训练与推理可以共用记录格式，但不能把 `step_time_ms` 与 `TTFT` 混成一个指标。

| 场景 | 优先指标 | 应拆开的阶段 |
|:---|:---|:---|
| 训练 | `step_time_ms`、samples/s | 数据读取、H2D、forward、backward、optimizer step |
| 单请求推理 | `TTFT`、`TPOT`、E2E latency | 输入处理、prefill、decode、流式输出 |
| 批处理 / 服务 | throughput、P50 / P99、queue wait | 排队、组批、执行、网络与输出 |

## 读 Trace 时区分三类事件

框架 operator、设备 kernel 与 host 侧事件位于不同层。operator 热点不自动等于 kernel 瓶颈；空白区间也要结合同步、输入搬运或上游依赖解释。

| 观察对象 | 可能含义 | 需要再核对的证据 |
|:---|:---|:---|
| operator 调用次数高 | 图被切得过碎或 Python 调度频繁 | kernel 数量、launch 间隙 |
| kernel 时长高 | 计算或访存可能成为热点 | shape、利用率、访存指标 |
| H2D / 数据读取拉长 | 输入管线进入关键路径 | DataLoader、预取、存储与 batch 形状 |
| CPU/GPU 空洞 | 同步、等待或依赖阻塞 | stream、队列与上游事件 |

![时间、输入搬运与关键路径](../../public/topic_discussion/performance_analysis/time_breakdown_trace.svg)

机制练习从 [Part 02 · PERF-TRACE Trace 依赖与关键路径](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.md)进入：先把事件、资源和前驱规范化，再区分事件总耗时、关键路径与暴露等待。

## 项目出口与相关阅读

[Part00 · 17 Profiling 基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.md) 提供框架级 trace；时间线确认候选后，可进入 [性能优化 Task4](../performance_optimization/intro.md) 判断计算、通信或 I/O 的策略方向。

相关阅读：[PyTorch Profiler schedule](https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.schedule)。
