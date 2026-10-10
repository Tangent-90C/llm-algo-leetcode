# 02 从归因证据到策略假设

## 页面目标

Task0 固定优化目标和约束；性能分析专题负责采集计时、profile、trace，完成瓶颈归因与回归口径。本页只消费这些结果，把阶段、热点、等待和驻留对象转成可比较的策略假设：下一步应进入训练状态、推理状态、关键路径还是成本准入。

## 核心机制：从证据到策略对象

| 已有证据 | 能确定什么 | 应形成的策略假设 | 仍不能直接推出 |
|:---|:---|:---|:---|
| 阶段计时稳定变慢 | 问题可复现，存在进一步比较的价值 | 训练状态、推理状态或输入/通信候选 | 不知道具体对象为何变慢 |
| profiler 显示热点 | 某段计算或调用值得纳入关键路径检查 | 算子路径、布局、融合或批次候选 | 热点一定决定端到端时间 |
| 显存时间线出现重叠 | 峰值来自一组同时驻留的对象 | checkpoint、offload、分片、容量预算候选 | reserved 等于活跃张量，或策略必然有效 |
| trace 出现空闲或长等待 | 依赖、通信或搬运可能暴露在关键路径 | 预取、分池、流、bucket 或 overlap 候选 | 时间线相交就已经产生有效 overlap |

工具的采集方法、同步要求和回归验证由[性能分析专题](../performance_analysis/intro.md)展开。本页只要求每条证据都落到一个可证伪的策略假设，避免把热点名称直接当作方案。

## 判断框架：热点不是结论

| 观察 | 可以提出的假设 | 仍需补的证据 |
|:---|:---|:---|
| 一个 kernel 占比高 | 计算或访存可能限制阶段时间 | 算术强度、带宽、调用次数与关键路径位置 |
| GPU 空闲较多 | CPU、I/O、同步或通信可能阻塞提交 | CPU 线程、H2D、collective、队列时间线 |
| 峰值显存靠近上限 | 某类状态的生命周期可能重叠 | 参数、梯度、activation、workspace 与 cache 账本 |
| 通信片段很长 | collective 或负载不均可能暴露 | 拓扑、transfer bytes、overlap 窗口与扩展效率 |

本页的输出应是一条可证伪表述，例如“在固定 workload 下，backward 的 activation 驻留同时抬高 peak memory 与 step time；下一步比较 checkpoint 的重算代价”。策略取舍由 Task2–Task5 完成。

![从归因证据到候选假设](../../public/topic_discussion/performance_optimization/attribution_to_candidate_hypothesis.svg)

## 连接到 Part00–02

| 作用 | 入口 | 本页如何使用 |
|:---|:---|:---|
| profiling 最小实践 | [Part00 · 17 Profiling](../../00_Prerequisites/17_PyTorch_Profiling_Basics.md)、[Part01 · 13 瓶颈分析](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) | 统一 warmup、同步、CPU / CUDA 时间与热点读法 |
| 训练 trace | [Part02 · 73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)、[74 Profiling 驱动优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | 将 trace 转成候选策略与复测记录 |
| 通信证据 | [Part02 · 47 NCCL 通信 Profiling](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md)、[48 通信热点](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md) | 判断 collective 是否位于关键路径 |

## 项目出口

完成本页后，应输出一份候选假设记录：归因证据、目标对象、候选机制、预期收益、可能转移的代价、证伪条件，以及下一步应进入的 Task。原始 profiling 记录继续由性能分析专题维护；真实优化收益必须回到非 profiler 的重复测量确认。
