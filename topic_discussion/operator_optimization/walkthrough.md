# 算子优化问题链：从计算内核到端到端收益

这条问题链假设你接手了一个可以运行、但性能不稳定的算子。目标不是直接寻找“最快写法”，而是逐步确认：语义是否正确、瓶颈是否明确、局部收益能否传到模型工作负载。

![算子优化：从语义到端到端验证](../../docs/public/topic_discussion/operator_optimization/operator_optimization_overview.svg)

## 候选计算内核怎样才算正确

从一个简单算子或模型组件开始，明确输入形状、数据类型、步长、边界和输出误差。Part 03 的[向量加法](../../03_Triton_Kernels/01_Triton_Vector_Addition.ipynb)和 [GEMM](../../03_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb)适合作为入口；参考实现先用 PyTorch 或 CPU 版本固定语义。

此时只回答一个问题：候选计算内核在什么条件下算“正确”？不要先用 GPU 时间替代正确性测试。

## 时间花在计算还是数据搬运

当语义契约稳定后，观察 load、计算和 store 如何映射到 global memory、L2、shared memory、register 或 Tensor Core。Part 01 的 [Warp / Block / Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb) 和 Part 03 的 [Triton 内存模型](../../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.ipynb)用于解释布局与 tile。

这里的判断是：候选实现减少了哪类搬运，代价又转移到了哪里？

## 图变换在什么条件下成立

在融合前先看依赖图：中间张量是否只有一个消费者、是否存在共享分支、布局转换或动态形状，以及目标执行后端能否表达改写后的计算。常量折叠、公共子表达式消除、节点重排和融合都是候选；它们只有在语义、布局与执行约束同时满足时才进入下一段。

此时要留下一个最小图级记录：改写前后依赖、被消除的读写、需要保留的同步点，以及不满足时的回退路径。图节点变少不是性能结论。

## 减少中间写回是否值得融合

如果多个阶段反复写回中间张量，再比较 RMSNorm、Softmax 或 Attention 的融合方案：[融合 RMSNorm](../../03_Triton_Kernels/03_Triton_Fused_RMSNorm.ipynb)、[融合 Softmax](../../03_Triton_Kernels/06_Triton_Fused_Softmax.ipynb)和 [FlashAttention](../../03_Triton_Kernels/08_Triton_Flash_Attention.ipynb)可以作为不同复杂度的例子。

融合前先列出依赖和中间张量，融合后再检查寄存器、共享内存、占用率和边界处理。减少计算内核数量不是充分条件。

## 同一配置为什么会因硬件而变化

同一个分块或融合方案在不同 GPU 上可能有不同结果。使用 Part 01 的 [Tensor Core](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.ipynb)，以及 Part 04 的 [CUDA / Triton / PyTorch 对照](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.ipynb)和 [CUDA Stream](../../04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer.ipynb)，把数据类型、形状、同步和异步搬运纳入解释。

此时才开始做 GPU 微基准，并固定预热、重复次数、GPU 和编译目标。

## 配置搜索怎样验证瓶颈假设

在多种形状或多个 GPU 上，手工选一个配置通常不够。进入 [Triton 自动调优与性能剖析](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb)，比较候选分块、warp 数和流水阶段数，并记录编译时间、缓存命中与运行时间。

性能剖析的作用是验证瓶颈假设：如果计算内核变快但端到端不变，应该检查启动、同步和其他阶段，而不是继续盲目调整分块。

## 局部收益能否传递到模型工作负载

最后用 [Triton Llama3 Block 项目](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb)把算子放回模型组件，再用固定工作负载比较端到端延迟、吞吐、显存、编译成本和质量。

项目报告至少保留：

- 参考实现与候选计算内核的语义和数值对照；
- 形状、数据类型、GPU、编译目标、预热和重复次数；
- 计算内核微基准与端到端基准；
- 性能剖析观察到的瓶颈和失败条件；
- `accept / tune / reject` 及其证据等级。

局部计算内核的收益只有在目标工作负载中重复出现，才可以成为端到端优化结论。
