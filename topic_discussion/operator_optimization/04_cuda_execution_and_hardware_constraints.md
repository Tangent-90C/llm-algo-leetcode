# 04. CUDA 执行与硬件约束

## 页面目标

本页把 kernel 配置和 GPU 执行资源联系起来：warp、block、shared memory、register、Tensor Core、stream 和异步搬运会共同决定实际执行时间。

![Kernel 执行路径：dtype、并行层次与异步搬运](../../docs/public/topic_discussion/operator_optimization/operator_execution_path.svg)

## 执行层次

| 层次 | 作用 | 主要约束 |
|:---|:---|:---|
| Thread / Warp | 执行标量或向量指令 | warp 利用率、分支和 mask |
| Block | 组织线程与 shared memory | block 大小、同步和资源上限 |
| Tensor Core | 执行适配的矩阵乘加 | dtype、shape 对齐、累加路径 |
| Stream / Event | 组织搬运、计算和依赖 | 可重叠性、同步点、空转 |

## 为什么同一 kernel 在不同 GPU 上不同

性能取决于硬件架构、SM 资源、内存层级、编译目标、dtype 和输入 shape。FLOPs、源代码长度或单个 kernel 的局部时间都不能单独代表端到端收益。

图级候选进入 backend 后，还会受到算子覆盖、layout 转换、运行时 fallback、launch 粒度与 stream 依赖限制。同一份 graph rewrite 在 CUDA、Triton 或不同 runtime 上可以走不同执行路径，因此 backend 约束应在决定候选时记录，而不是在结果不一致后才补充解释。

## 累加精度与执行路径

输入 dtype 不等于整个计算过程的 dtype。FP16、BF16 或 FP8 输入可能使用 FP32 累加器；Softmax、Norm 和 Reduction 尤其需要关注累加顺序、scale 和数值稳定性。报告中应分别记录输入 dtype、累加 dtype 和输出 dtype。

| 观察点 | 需要确认 | 影响 |
|:---|:---|:---|
| 输入 dtype | FP32、FP16、BF16 或 FP8 | 存储量、Tensor Core 候选路径 |
| 累加 dtype | FP16 / BF16 还是 FP32 | 误差、溢出和吞吐 |
| shape 对齐 | 矩阵维度是否满足指令要求 | 是否能进入特定指令路径 |
| 同步与重叠 | copy、compute、通信是否可重叠 | stream 空转和端到端延迟 |

## 从框架调用到 Kernel

一个算子通常经过 `PyTorch eager → torch.compile / Inductor → Triton 或 CUDA kernel → GPU runtime`。图变换、IR、lowering 与自动选择属于这条执行链的前半段；写出候选 kernel 后，还要确认模型或 Block 是否真的调用了它。

| 实验层级 | 观察内容 | 适合回答的问题 |
|:---|:---|:---|
| kernel microbenchmark | CUDA 时间、吞吐、显存 | 候选 kernel 是否更快？ |
| 编译与调度 | 编译时间、launch、stream | 优化成本是否可接受？ |
| workload benchmark | 端到端延迟、吞吐、显存 | 局部收益是否传递到系统？ |

## 学习顺序

Part 01 的 Tensor Core 建立硬件机制，Part 04 的 MMA、Warp primitive、Shared Memory、CUDA Stream 和 CUDA / Triton / PyTorch 对照用于实际执行验证。先固定 GPU、dtype 和 shape，再比较配置，避免把硬件差异误认为算法收益。

## 本页出口

你应能说明一个配置为什么可能适合某张 GPU，指出它依赖的 dtype、shape、资源和同步条件，并设计分层 benchmark 验证它。
