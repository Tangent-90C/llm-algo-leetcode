# 08. Programming Models CUDA Triton | 编程模型演进

**难度：** Medium | **环境：** GPU optional

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `算子编程`, `CUDA`, `Triton` | **目标人群：** 编译与系统入门者

---

## 本节导读

从 PyTorch 写法走到 CUDA / Triton，并不是为了“更底层”本身，而是因为框架级算子组合并不总能贴住硬件执行粒度。只要中间张量太多、launch 太碎，或者数据布局和 tile 组织没法显式控制，性能上限就会很快被访存和调度开销卡住。

这一页在整个教程的纵向主线里属于 `Part 01` 通向 `Part 03 / Part 04` 的桥接页，优先服务 `推理优化路线` 里的算子下沉判断，也给后续系统优化页建立编程模型前置。学完这里，后面再看 `18 / 19 / 32` 以及 `Part 03` 的 Triton kernel 实战时，你会更容易判断瓶颈到底停留在框架级组合、算子级组织，还是已经值得下沉到 CUDA / Triton；如果这里没学明白，后面很容易把“自定义 kernel”误当成默认更高级的选择，而说不清它到底补的是哪一层控制权。按专题归类，这一页主要属于 `编译与图优化专题` 的基础入口，同时支撑后续 kernel 实现主线。

**关键词：** `grid`, `block`, `kernel`

---
## 前置阅读
**导语：** 这一页先把从 PyTorch 到 CUDA / Triton 的编程模型演进接上，再看为什么自定义算子会更接近硬件。
- [Group 1D: Execution Models, Asynchrony and Operator Mapping | 1D: 执行模型、异步调度与算子映射](./1D.md)
- [15. CUDA Execution Model | CUDA 执行模型](./15_CUDA_Execution_Model.md)
- [16. Warp Block SharedMemory Basics | Warp、Block 与 Shared Memory 基础](./16_Warp_Block_SharedMemory_Basics.md)
## 相关阅读
**导语：** 如果想继续把编程模型和 kernel 组织方式补完整，可以接着看这些页。
- [Part 03: Triton Kernel Development | 第三部分：Triton 算子开发](../03_Triton_Kernels/intro.md)
- [01. Triton 入门与 Hello World：向量加法 (Vector Addition)](../03_Triton_Kernels/01_Triton_Vector_Addition.md)
- [04. Triton 矩阵乘法 (GEMM) 与自动调优 (Autotune)](../03_Triton_Kernels/04_Triton_GEMM_Tutorial.md)
---
## Q1：为什么我们需要用 CUDA 或 Triton 编写自定义算子？

<details><summary>点击展开查看解析</summary>

PyTorch 原生操作方便，但它默认把很多执行细节交给框架和后端。

当一个计算链里出现大量小算子、重复的 HBM 往返、或者对布局和 tile 有强约束时，原生写法往往会留下更多中间张量和 launch 开销。自定义 CUDA 或 Triton 的价值，不是“更底层”本身，而是把算子组织方式重新对齐到硬件执行粒度。

所以这类问题的关键，不是“要不要手写 kernel”，而是“哪些数据流和执行流必须由我们显式控制”。
</details>
### Q1小验证：为什么原生操作可能慢

先想是不是中间张量太多。

```python
def memory_traffic(num_ops, fused=False):
    # 自定义算子的价值，不在于把名字写得更底层，而在于减少中间张量落地。
    input_reads = num_ops
    output_writes = 1
    intermediate_writes = max(num_ops - 1, 0) if not fused else max(num_ops - 2, 0)
    return input_reads + output_writes + intermediate_writes

cases = [
    ('separate_ops', memory_traffic(3, fused=False)),
    ('fused_ops', memory_traffic(3, fused=True)),
    ('longer_chain', memory_traffic(5, fused=False)),
]
for name, traffic in cases:
    print(name, '-> traffic units:', traffic)
print('fusion wins when it removes intermediate writes, not when it only shortens code')

```

## Q2：CUDA 的线程层级结构与 GPU 硬件执行单元如何对应？

<details><summary>点击展开查看解析</summary>

CUDA 的层级不是语法装饰，而是把“任务怎么切”映射到“硬件怎么调度”。

- grid 决定一次 kernel launch 覆盖多少独立工作；
- block 决定哪一组线程能共享数据和同步；
- warp 是硬件实际调度的基本单位；
- thread 是最细的计算粒度。

理解这层映射后，就能看出为什么同样的代码，线程数、block 数和 warp 利用率不同，性能会差很多。
</details>
### Q2小验证：层级怎么对应

把执行层级记清楚，后面读 kernel 会更顺。

```python
def block_launch_stats(num_threads, warp_size=32):
    # grid/block/warp/thread 的对应关系，最终会落到 warp 数和空转 lane 上。
    warps = (num_threads + warp_size - 1) // warp_size
    active_lanes = num_threads
    wasted_lanes = warps * warp_size - num_threads
    occupancy = active_lanes / (warps * warp_size)
    return {'warps': warps, 'wasted_lanes': wasted_lanes, 'occupancy': round(occupancy, 2)}

for threads in [48, 64, 96, 128]:
    print(threads, 'threads ->', block_launch_stats(threads))
print('线程层级的关键不是层级名，而是 warp 是否被填满')

```

## Q3：跨线程共享和同步机制如何限制和优化性能？

<details><summary>点击展开查看解析</summary>

跨线程共享真正解决的是“同一份数据能不能在片上被复用”。

如果数据只被读一次，shared memory 和同步的开销可能不值；如果同一份数据会被 block 内多个线程反复使用，先放到 shared memory 再同步，通常能换回更多 HBM 访问的节省。

所以这里不是“共享越多越好”，而是“复用收益是否足以覆盖同步代价”。
</details>
### Q3小验证：共享为什么会和同步绑定

复用越多，协作就越重要。

```python
def shared_tradeoff(reuse_times, sync_points, hbm_cost=10, smem_cost=2, sync_cost=3):
    # 共享只有在复用收益大于同步代价时才值得。
    saved_hbm = max(reuse_times - 1, 0) * (hbm_cost - smem_cost)
    penalty = sync_points * sync_cost
    return {'net_gain': saved_hbm - penalty, 'worth_it': saved_hbm > penalty}

for case in [(1, 0), (2, 1), (4, 1), (4, 3)]:
    print(case, '->', shared_tradeoff(*case))
print('shared memory is good only when reuse amortizes synchronization')

```

## Q4：Triton 为什么能降低算子开发门槛？

<details><summary>点击展开查看解析</summary>

Triton 的抽象重点，不是“帮你自动变快”，而是把 tile、block 和布局这些样板性细节收起来，让你先把张量切分方式写清楚。

和 CUDA 相比，它并不是取消底层约束，而是把底层约束收束成更少的显式概念：布局、mask、tile 和 program 关系。

所以 Triton 的门槛更低，是因为它减少了模板代码；但它的性能边界，仍然取决于你对执行粒度和数据布局的理解。
</details>
### Q4小验证：Triton 抽象了什么

先看块组织，再看算子细节。

```python
def triton_benefit(tile, layout_contiguous=True, boilerplate_lines=80):
    # Triton 的门槛低，靠的是把 tile / layout / launch 的样板代码抽掉。
    if not layout_contiguous:
        return {'boilerplate_saved': 0, 'ready': False}
    saved = max(boilerplate_lines - tile // 2, 0)
    return {'boilerplate_saved': saved, 'ready': True}

for case in [(64, True, 80), (128, True, 80), (128, False, 80)]:
    print(case, '->', triton_benefit(*case))
print('Triton abstracts launch and tiling details, but still depends on layout discipline')

```

## ⚠️ 常见误区

- 自定义算子不是为了炫技，而是为了更好地控制执行路径。
- CUDA 和 Triton 是不同范式，但目标都在靠近硬件。
- 共享和同步是一起看的，不能只谈一个。
- 编程模型的核心是映射，不是语法本身。