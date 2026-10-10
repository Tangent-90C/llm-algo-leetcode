# 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行

**难度：** Medium | **环境：** GPU optional | **标签：** `算子编程`, `CUDA Stream`, `异步执行` | **目标人群：** 编译与系统入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

吞吐上不去，很多时候不是单个 kernel 太慢，而是 kernel、拷贝和同步被排成了一条过于保守的串行链。只要不同任务之间本来可以重叠，却因为 stream 使用方式不当而互相等待，GPU 的空转时间就会很快被放大。

这一页在整个教程的纵向主线里属于 `Part 01` 的调度与重叠基础页，优先服务 `推理优化路线` 的吞吐判断，也给后续 CUDA 执行页建立异步调度前置。学完这里，后面再看 `29` 以及 `66 / 67` 这类 benchmark 和部署页时，你会更容易判断问题出在“算得慢”还是“排得差”；如果这里没学明白，后面很容易把 stream、异步和 overlap 只当成几条 API，而说不清同步点、拷贝和 kernel 排队到底是怎样把 GPU 空转时间放大的。按专题归类，这一页主要属于 `推理优化专题` 的系统执行前置，也和 `Profiling 专题`、`编译与图优化专题` 共享调度视角。

**关键词：** `stream`, `asynchrony`, `overlap`

---
## 前置阅读

**导语：** 先把 block 和 shared memory 的关系对齐，再看 stream 和异步重叠会更顺。

- [Group 1C: Communication, Parallelism and Fault Tolerance | 1C: 通信、并行与容错](./1C.md)
- [15. CUDA Execution Model | CUDA 执行模型](./15_CUDA_Execution_Model.md)
- [16. Warp Block SharedMemory Basics | Warp、Block 与 Shared Memory 基础](./16_Warp_Block_SharedMemory_Basics.md)

## 相关阅读

**导语：** 把 stream 放进 profiling、缓存访问和调度视角里看，更容易判断吞吐瓶颈是怎么形成的。

- [05. Triton 性能调优与基准测试 (Autotune & Profiling)](../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md)
- [09. Triton PagedAttention | KV Cache 间接寻址](../03_Triton_Kernels/09_Triton_PagedAttention.md)
- [12. Triton Memory Model and Debug | 内存模型、指针计算与 Debug 避坑指南](../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.md)
---
## Q1：CUDA Stream 到底是什么，为什么它不是“自动加速器”？

<details>
<summary>点击展开查看解析</summary>

CUDA Stream 本质上是任务的顺序队列，用来告诉 GPU 哪些操作按什么顺序执行。

它不是魔法加速器，因为 stream 本身不会让某个 kernel 变快；它的价值在于**组织顺序**和**给并行重叠创造条件**。

如果一个程序里所有任务都排在同一条队列上，很多事情就只能串行执行。只有把任务拆到不同 stream，或者把拷贝、计算、通信放进可重叠的阶段，才有机会把空转时间压下去。
</details>
### Q1小验证：stream 解决的是什么问题

先把“队列”这个概念记住，再看重叠就容易多了。

```python
def stream_role(copy_ms, compute_ms, sync_ms):
    # Stream 的本质不是名字，而是能否把搬运、计算和同步的时间重叠起来。
    serial = copy_ms + compute_ms + sync_ms
    overlap = max(copy_ms, compute_ms) + sync_ms
    return serial, overlap

cases = [(10, 20, 4), (4, 20, 4), (12, 8, 2)]
for case in cases:
    serial, overlap = stream_role(*case)
    print(case, '->', {'serial': serial, 'overlap': overlap, 'speedup': round(serial / overlap, 2)})
print('stream helps only when copy / compute / sync can overlap')

```

## Q2：为什么异步执行的核心是 overlap，而不是“多开几个任务”？

<details>
<summary>点击展开查看解析</summary>

异步执行的收益不在于“任务更多”，而在于**把原本会等待的时间藏起来**。

常见的重叠包括：
- 数据搬运和计算重叠；
- 前一个 kernel 和后一个 kernel 重叠；
- 通信和计算重叠；
- pipeline 的不同阶段重叠。

如果这些阶段彼此完全依赖，就算开再多 stream 也没意义。真正有价值的异步，是把本来空着的硬件资源填起来，让 GPU 不必在搬运或同步上干等。
</details>
### Q2小验证：什么叫 overlap

先分清“并发启动”和“真正重叠”不是一回事。

```python
def overlap_hint(copy_time, compute_time, sync_points=1):
    # overlap 不是 yes/no，而是搬运和计算是否都足够长，能抵消同步代价。
    gain = max(copy_time, compute_time) + sync_points - (copy_time + compute_time + sync_points)
    return {'gain': gain, 'worth_it': gain < 0}

for case in [(10, 20, 1), (2, 40, 1), (1, 3, 3)]:
    print(case, '->', overlap_hint(*case))
print('small copy or compute phases are harder to overlap effectively')

```

## Q3：为什么调度问题经常比单个算子更影响吞吐？

<details>
<summary>点击展开查看解析</summary>

很多系统慢，不是因为某个算子本身特别差，而是因为调度让大量时间花在等待、同步和小碎操作上。

典型场景包括：
- kernel 太碎，launch 开销被放大；
- 数据搬运没有和计算重叠；
- 任务之间同步太频繁；
- stream 组织不合理，导致本可以并行的阶段被串起来。

所以看吞吐问题时，不能只盯着单个 kernel 的速度，还要看整个执行链是不是被调度拖住了。
</details>
### Q3小验证：先判断是不是调度问题

如果单个算子不慢，但整体吞吐差，就先怀疑调度。

```python
def bottleneck_style(single_kernel_fast=True, whole_pipeline_slow=False, sync_points=0):
    # 单算子快不代表整条流水线快，调度和同步同样会成为瓶颈。
    score = 0
    score += 2 if whole_pipeline_slow else 0
    score += 1 if sync_points > 2 else 0
    score += 1 if not single_kernel_fast else 0
    return score

plans = [
    ('kernel_bound', True, False, 0),
    ('schedule_bound', True, True, 3),
    ('mixed', False, True, 2),
]
for name, fast, slow, sync_points in plans:
    print(name, '-> bottleneck score', bottleneck_style(fast, slow, sync_points))
print('pipeline-level bottlenecks are not visible from one kernel alone')

```

## ⚠️ 常见误区

- stream 不是越多越好，关键是任务是否真的能重叠。
- 异步执行不是取消等待，而是把等待藏到别的工作里。
- 吞吐差不一定是 kernel 慢，也可能是调度方式有问题。
- 先想 overlap，再想优化代码，通常更接近问题本质。