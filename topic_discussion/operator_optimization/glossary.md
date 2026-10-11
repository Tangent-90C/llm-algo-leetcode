# 算子优化专题术语表

本表解释从参考语义到计算内核和端到端复测时容易混淆的概念。全教程统一写法见[术语表与中英文写作规范](../../docs/terminology.md)。

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 参考实现 | reference implementation | 用于固定输入输出语义和数值容差的可读实现 | 被优化的候选实现 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) |
| 计算内核 | kernel | 在加速器上执行一段并行计算的程序单元 | 操作系统内核 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) |
| 分块 | tiling | 把输入域切成适合线程组织和片上存储的工作块 | 仅改变张量形状 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) |
| 合并访存 | memory coalescing | 让相邻线程的访问形成较少且连续的显存事务 | 缓存命中率 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) |
| 片上复用 | on-chip reuse | 在寄存器或共享内存中复用已加载数据，减少全局显存访问 | 主机内存缓存 | [03 图级优化、融合与计算内核组合](./03_fusion_and_kernel_composition.md) |
| 算子融合 | operator fusion | 在语义允许时把多个操作放入同一执行单元，减少中间写回与启动 | 只减少图节点数量 | [03 图级优化、融合与计算内核组合](./03_fusion_and_kernel_composition.md) |
| 占用率 | occupancy | 活跃线程束相对硬件上限的比例，是资源约束指标之一 | GPU 总利用率 | [04 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) |
| 启动开销 | launch overhead | 主机提交一次计算内核及运行时调度产生的固定成本 | 计算内核内部计算时间 | [01 算子性能心智模型](./01_why_operator_optimization_matters.md) |
| 自动调优 | autotuning | 在固定语义和输入分布下搜索分块、线程束或流水阶段配置 | 不受约束地尝试任意代码 | [05 成本模型与性能剖析](./05_cost_model_and_profiling.md) |
| 回退路径 | fallback path | 目标后端或计算内核不满足条件时实际采用的替代执行路径 | 正常命中目标路径 | [03 图级优化、融合与计算内核组合](./03_fusion_and_kernel_composition.md) |
| 微基准 | microbenchmark | 在固定输入下隔离测量单个计算内核或局部组合 | 模型端到端结果 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |
| 端到端复测 | end-to-end validation | 把局部候选放回模型或服务工作负载，检查收益与守护条件 | 用微基准推断整体收益 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |

代码标识、API、配置键和工具名称保留原文；普通叙述优先使用中文主称。性能结论必须同时说明输入形状、布局、数据类型、设备、后端、预热和重复协议。
