# 03 时间线、I/O 与关键路径

同样是“慢”，可能来自计算量、计算内核启动、输入搬运、同步等待或请求排队。先建立基线，再用阶段时间线判断哪一段进入关键路径。

## 建立可比较的时间基线

先固定模型、输入、精度、预热、重复次数与同步方式；再选择与问题匹配的指标。训练与推理可以共用记录格式，但不能把 `step_time_ms` 与 `TTFT` 混成一个指标。

| 场景 | 优先指标 | 应拆开的阶段 |
|:---|:---|:---|
| 训练 | `step_time_ms`、样本数/秒 | 数据读取、H2D、前向传播、反向传播、优化器更新 |
| 单请求推理 | `TTFT`、`TPOT`、端到端延迟 | 输入处理、预填充、解码、流式输出 |
| 批处理 / 服务 | 吞吐、P50 / P99、排队等待 | 排队、组批、执行、网络与输出 |

## 读执行轨迹时区分三类事件

框架算子、设备计算内核与主机侧事件位于不同层。算子热点不自动等于计算内核瓶颈；空白区间也要结合同步、输入搬运或上游依赖解释。

| 观察对象 | 可能含义 | 需要再核对的证据 |
|:---|:---|:---|
| 算子调用次数高 | 图被切得过碎或 Python 调度频繁 | 计算内核数量、启动间隙 |
| 计算内核时长高 | 计算或访存可能成为热点 | 形状、利用率、访存指标 |
| H2D / 数据读取拉长 | 输入管线进入关键路径 | DataLoader、预取、存储与 batch 形状 |
| CPU/GPU 空洞 | 同步、等待或依赖阻塞 | stream、队列与上游事件 |

![时间、输入搬运与关键路径](../../docs/public/topic_discussion/performance_analysis/time_breakdown_trace.svg)

机制练习从 [Part 02 · PERF-TRACE Trace 依赖与关键路径](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb)进入：先把事件、资源和前驱规范化，再区分事件总耗时、关键路径与暴露等待。

## 按分析目标选择工具层

工具选择服从要回答的问题，而不是越底层越好。先用轻量计时确认现象，再用框架追踪定位阶段；只有证据落到跨线程等待、单个计算内核或多进程通信时，才升级到系统或硬件级工具。

| 分析目标 | 首选工具与证据 | 升级条件 | 升级后的工具 |
|:---|:---|:---|:---|
| 确认总时间、阶段时间或峰值是否变化 | `perf_counter`、CUDA Event、显存 API 的重复样本 | 聚合指标无法解释时间去了哪里 | PyTorch Profiler |
| 定位算子、输入搬运和 CPU/GPU 空洞 | PyTorch Profiler 的事件、形状和执行轨迹 | 需要解释跨线程、CUDA 流、同步或系统 I/O | Nsight Systems |
| 解释单个计算内核效率 | 框架追踪确认该计算内核位于关键路径 | 需要访存吞吐、占用率和指令级证据 | Nsight Compute |
| 解释显存峰值 | PyTorch 显存统计、显存快照和对象驻留时间线 | 无法确认对象身份、分配栈或峰值重叠 | 显存快照与调用栈联合分析 |
| 解释多卡扩展损失 | 每进程阶段时间、载荷和集合通信记录 | 需要区分最慢进程、拓扑、负载不均与暴露通信 | NCCL 执行轨迹、HTA 或 Nsight Systems |
| 解释推理服务尾延迟 | TTFT、TPOT、队列和请求执行轨迹 | 需要关联缓存、批次、调度与后端执行 | 服务后端的请求追踪与系统时间线 |

每次升级都应保留上一层证据：系统追踪不能替代固定工作负载，计算内核报告也不能证明该内核位于端到端关键路径。TrainLens、`vllm-htop` 一类实时工具适合快速发现异常时段，但正式归因仍需导出可复查的执行轨迹、性能剖析报告或配对样本。

## 保留可复查的工具产物

不同工具的原始输出不能只截屏保存。结果记录至少要能回到本次运行身份，并保留后续重新解释所需的原始产物。

| 工具层 | 应保存的原始产物 | 同时记录的上下文 |
|:---|:---|:---|
| 轻量测量 | 每次计时、峰值和失败样本 | 单位、同步方式、预热与重复次数 |
| PyTorch Profiler | 执行轨迹文件与算子摘要 | schedule、活动类型、shape 与调用栈开关 |
| 显存分析 | 显存快照或对象驻留记录 | 峰值阶段、对象身份、allocated / reserved |
| Nsight Systems | 系统追踪报告 | 采集命令、时间窗口、进程与设备 |
| Nsight Compute | 计算内核报告 | 计算内核名称、输入形状、指标集合与调用位置 |
| NCCL / 多进程追踪 | 各进程执行轨迹与集合通信摘要 | world size、拓扑、载荷、最慢进程和时钟口径 |
| 服务追踪 | 请求级执行轨迹 | 到达时间、输入/输出长度、缓存、批次与调度状态 |

具体文件字段和登记位置见[性能分析实验资产表](./experiment_assets.md)。

## 进入对象与资源归因

本页继承 Task1 固定的基线。[Part 01 · 07 CPU/GPU 异构调度](../../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb)、[Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb)和[Part 01 · 17 CUDA 流](../../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb)分别提供搬运、启动、同步与异步执行基础；时间线确认候选后，进入[正文 04 显存时间线](./04_memory_timeline_and_residency.md)或[正文 05 瓶颈归因](./05_bottleneck_attribution.md)补足对象与资源证据。

相关阅读：[PyTorch Profiler schedule](https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.schedule)。
