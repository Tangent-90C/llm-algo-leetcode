# 04 推理状态容量与 KV Cache 成本

## 页面目标

推理侧的容量问题来自持续增长的状态，而不只是一次前向的临时张量。KV Cache 的表示、序列长度、并发请求、复用命中、驱逐策略和 dtype 共同决定能容纳多少活跃请求，以及 TTFT、TPOT 与质量会如何变化。

本页从性能证据视角衡量状态成本；PagedAttention、RadixAttention、Prefix Cache 或 MLA 的具体机制仍由推理优化与模型架构页面展开。

## 核心机制：状态容量如何影响服务指标

| 状态变化 | 容量收益 | 可能引入的代价 | 必须同时观察 |
|:---|:---|:---|:---|
| 减少 KV 表示或降低 dtype | 每 token 占用下降 | 误差、反量化、kernel/backend 限制 | peak memory、质量、TPOT |
| 分页分配与驱逐 | 降低碎片、控制活跃容量 | 页管理、重算或命中损失 | 并发、尾延迟、失败率 |
| 前缀复用 | 减少重复 Prefill 与状态创建 | 匹配与维护成本、命中分布依赖 | hit rate、TTFT、维护开销 |
| 调整并发或长度预算 | 避免超容量与 OOM | 排队、截断或服务质量退化 | queue wait、P99、质量 |

容量结论必须回到请求分布：同一策略在共享前缀很多的 workload 中可能显著改善 TTFT，在随机请求中却只有维护成本；低比特 KV 也不能仅凭账本估算宣称 backend 已加速。

| 容量字段 | 用途 | 必须配对的服务证据 |
|:---|:---|:---|
| 每 token 的 K/V 表示与 dtype | 估算单请求随上下文增长的基础成本 | 长度分布与 peak memory |
| 活跃请求数与并发上限 | 估算状态同时驻留的容量压力 | queue wait、P99 与拒绝/OOM 记录 |
| block 利用率、碎片与驱逐 | 判断物理分配是否浪费容量 | 分页管理开销与失败原因 |
| 命中率、复用 token 与失效原因 | 判断重复 Prefill 是否被避免 | TTFT、维护时间与请求分布 |

![KV 容量、并发与服务延迟关系](../../public/topic_discussion/performance_optimization/kv_capacity_concurrency_latency.svg)

## 判断框架：容量问题不是单一命中率问题

| 观察 | 先验证什么 | 后续候选 |
|:---|:---|:---|
| OOM 或并发突然下降 | 每 token KV 成本、活跃序列长度、页碎片和保留策略 | 长度/并发预算、分页、驱逐、表示压缩 |
| hit rate 高但 TTFT 没改善 | 匹配时间、维护开销、请求分布和 Prefill 占比 | 修改匹配粒度、路由或缓存准入 |
| 量化后显存下降但质量/P99 变差 | dtype、量化粒度、backend 路径和质量集 | 调整表示、回退精度或分层缓存 |

本页的输出是容量—延迟—质量的对照，不是某个缓存实现的采用结论；是否适合具体服务 workload 由 Task5/Task6 的 SLO 与成本证据决定。

## 连接到 Part01–02

| 作用 | 入口 | 本页如何使用 |
|:---|:---|:---|
| KV 容量公式 | [Part01 · 11 KV Cache 增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)、[Part02 · 22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) | 计算状态规模与物理分配 |
| 复用与调度 | [Part02 · 24 Prefix Cache](../../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.md)、[34 RadixAttention](../../02_PyTorch_Algorithms/34_SGLang_RadixAttention.md)、[69 前缀缓存基准](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md) | 测量命中、TTFT 与维护代价 |
| 表示压缩 | [Part02 · 41 FP8 运行时](../../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md)、[54 K/V 量化策略](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md)、[71 MLA 基准](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md) | 分开记录账本、质量与真实执行后端证据 |

## 项目出口

完成本页后，应形成一份推理状态容量表：workload 的长度/并发/共享前缀分布，baseline/candidate 的每 token 成本、命中率、peak memory、TTFT、TPOT、P99、质量和 evidence level。进入 Task4 后，再判断瓶颈是否已从状态容量转移到计算、通信或 I/O。
