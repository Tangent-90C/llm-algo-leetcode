# 05 瓶颈归因：计算、带宽、容量、通信、I/O 与排队

执行轨迹或性能剖析给出的是事件、热点和驻留对象；瓶颈归因要进一步解释，哪一类资源约束正在限制端到端指标。归因必须能够被下一组对照推翻，不能只根据一个耗时占比给优化方案。

## 从现象建立可证伪归因

先把关键路径上的现象映射到候选瓶颈，再指定下一项判别证据。多个类别可以同时存在，但第一轮只选择最可能限制目标指标的一项验证。

| 瓶颈类别 | 常见现象 | 下一项判别证据 | 不应直接推出 |
|:---|:---|:---|:---|
| 计算 | 核心算子持续占据关键路径 | FLOPs、Tensor Core 路径、计算内核时间与调用次数 | 算子耗时占比高就必须重写计算内核 |
| 带宽 | 数据量增加时延迟近似线性增长 | bytes、带宽利用率、算术强度和缓存命中 | GPU 利用率低就是显存带宽受限 |
| 容量 | OOM、并发下降或频繁驱逐 | 活跃对象、峰值驻留、碎片与回收事件 | reserved memory 等于有效状态量 |
| 通信 | 集合通信或同步暴露在关键路径 | 载荷、频率、每进程时间、拓扑和重叠窗口 | 多卡扩展差只由网络带宽造成 |
| I/O | GPU 等待输入、产物或状态搬运 | 读取/解析/H2D/回写阶段时间与队列深度 | 文件读取慢就是端到端主瓶颈 |
| 排队 | TTFT/P99 上升但单次计算稳定 | 到达轨迹、queue wait、准入和批次状态 | 增大 batch 一定提高服务效率 |

[Part 01 · 02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb)给出理论计算量，[Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)给出计算、带宽与容量上界，[Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)建立归因方法。[Part 02 · PERF-ROOFLINE 算术强度与资源上界](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb)再将 FLOPs、最低搬运字节、经验计算吞吐和经验带宽放到同一份账本中，用于区分计算与带宽候选；端到端归因仍需关键路径证据。

## 通信暴露是归因案例，不是独立路线

多卡吞吐未按卡数增长时，应把集合通信、同步、数据搬运与计算放到同一时间线上。通信持续时间只是原始成本；只有没有被计算隐藏的部分进入关键路径时，才构成暴露通信瓶颈。

| 观察 | 可能原因 | 对照方式 |
|:---|:---|:---|
| 集合通信持续时间升高 | 载荷、拓扑或调用频率变化 | 固定模型和全局批量，对齐消息字节与集合通信次数 |
| 计算前后出现空洞 | 同步、依赖或发起时机 | 查看 CUDA 流、事件依赖和每进程时间线 |
| 重叠窗口缩小 | 可重叠计算不足或调度次序变化 | 比较通信时间、可重叠窗口和暴露时间 |
| 尾部进程持续落后 | 数据、路由或硬件负载不均 | 记录每进程阶段时间与载荷分布 |

![通信等待与计算重叠](../../docs/public/topic_discussion/performance_analysis/communication_overlap_map.svg)

## 从归因进入判别实验

本页输出“目标指标 → 关键路径事件 → 瓶颈类别 → 判别实验”，再交给性能优化选择候选。通信分支从 [Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb)和 [Part 01 · 20 NCCL 与 AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)建立载荷与拓扑账本，再从 [Part 02 · 47 NCCL 通信性能分析](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb)取得真实执行轨迹。

相关阅读：[PyTorch Profiler 文档](https://docs.pytorch.org/docs/stable/profiler.html)、[NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)。
