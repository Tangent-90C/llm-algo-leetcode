# 算子优化判断手册

当候选实现已经能运行，但正确性、局部性能或端到端收益不符合预期时，先固定输入形状、布局、数据类型、设备、编译目标和测量协议。至少记录参考输出、误差、预热与重复样本、实际执行路径以及失败输入。

从“现象”列找到最接近的一行，按“先检查什么”采集最小证据；证据不足时不要直接改分块大小、融合范围或自动调优配置。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 输出形状正确但数值误差超限 | 数据类型、掩码、边界分块和累加路径 | 参考输出、最大误差、失败索引与边界输入 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) | [Part 03 · 12 Triton 内存模型与调试](../../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.ipynb) |
| 尾部分块越界或遗漏元素 | 程序 ID、有效范围和加载/写回掩码 | 非整除形状、访问范围与逐元素对照 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) | [Part 03 · 01 Triton 向量加法](../../03_Triton_Kernels/01_Triton_Vector_Addition.ipynb) |
| 计算内核能运行但明显慢于参考实现 | 合并访存、布局、分块工作集和启动次数 | 固定形状的配对延迟、字节量与计算内核轨迹 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) | [Part 03 · 05 自动调优与性能剖析](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb) |
| 融合后计算内核更慢 | 寄存器、共享内存、占用率和重复计算 | 融合前后中间读写、资源用量与配对延迟 | [03 图级优化、融合与计算内核组合](./03_fusion_and_kernel_composition.md) | [Part 01 · 19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb) |
| 图已改写但没有进入目标计算内核 | 后端支持、布局转换、动态形状与回退路径 | 编译图、实际计算内核名称和回退原因 | [03 图级优化、融合与计算内核组合](./03_fusion_and_kernel_composition.md) | [性能分析专题](../performance_analysis/intro.md) |
| 同一实现换到另一张 GPU 后退化 | 编译目标、数据类型能力、分块资源和同步方式 | 两台设备的硬件信息、编译配置与同口径样本 | [04 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) | [Part 04 · 21 CUDA / Triton / PyTorch](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.ipynb) |
| 自动调优结果在重复运行间变化 | 形状分布、预热、编译缓存和候选搜索空间 | 每个候选的重复样本、缓存状态和获胜配置 | [05 成本模型与性能剖析](./05_cost_model_and_profiling.md) | [Part 03 · 05 自动调优与性能剖析](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb) |
| 微基准变快但模型工作负载不变 | 局部计算内核占比、启动、同步和相邻阶段 | 局部收益、端到端配对结果与关键路径轨迹 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) | [性能优化专题](../performance_optimization/intro.md) |
| 低比特计算内核更快但质量下降 | 量化对象、缩放、累加精度和回退路径 | 质量结果、实际数据类型、计算内核与回退记录 | [04 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) | [量化与压缩专题](../quantization_compression/intro.md) |
| 多卡作业中本地计算内核不是关键路径 | 暴露通信、最慢进程和计算通信重叠 | 各进程时间线、集合通信载荷与暴露时间 | [05 成本模型与性能剖析](./05_cost_model_and_profiling.md) | [通信与并行专题](../communication_parallelism/intro.md) |
