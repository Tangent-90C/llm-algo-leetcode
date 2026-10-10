# 19. Operator Fusion Introduction | 算子融合导论

**难度：** Medium | **环境：** CPU-first | **标签：** `编译系统`, `算子融合`, `性能优化` | **目标人群：** 编译与系统入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

算子融合真正关心的，不是把几个名字并成一个名字，而是减少中间张量落地、减少 HBM 往返，以及把原本被拆碎的执行路径重新贴回硬件粒度。很多性能损失，并不是单个算子慢，而是算子之间来回搬数据太多。

这一页在整个教程的纵向主线里属于 `Part 01` 从执行模型走向图优化的桥接页，优先服务后续 `编译与图优化专题`，也给 `推理优化路线` 补一层数据流视角。学完这里，后面再看 `30 / 32` 以及 `Part 03 / Part 04` 的 kernel 与 backend 优化页时，你会更容易判断“该不该融合、为什么融合会快”；如果这里没学明白，后面很容易把 fusion 只当成编译器关键词，而说不清中间张量写回、调试成本和形状复杂度为什么会一起决定融合值不值得做。按专题归类，这一页同时属于 `编译与图优化专题`，并和 `推理优化专题` 共享执行链路视角。

**关键词：** `fusion`, `memory traffic`, `intermediate tensor`

---
## 前置阅读

**导语：** 先把执行模型和 block 级编程对齐，再看算子融合为什么能减少中间结果开销会更顺。

- [Group 1D: Execution Models, Asynchrony and Operator Mapping | 1D: 执行模型、异步调度与算子映射](./1D.md)
- [Group 1E: Compilers and Backends | 1E: 编译与后端](./1E.md)
- [18. Triton Block Model | Triton Block 模型](./18_Triton_Block_Model.md)

## 相关阅读

**导语：** 把算子融合继续接到性能调优、归约和 attention kernel，会更直观地看到中间张量是怎么被省掉的。

- [05. Triton 性能调优与基准测试 (Autotune & Profiling)](../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md)
- [06. Triton 进阶：跨线程归约与数值稳定 (Safe Softmax)](../03_Triton_Kernels/06_Triton_Fused_Softmax.md)
- [08. Triton Flash Attention | 真正的 Flash Attention 前向算子](../03_Triton_Kernels/08_Triton_Flash_Attention.md)
---
## Q1：为什么算子之间的中间结果会这么贵？

<details>
<summary>点击展开查看解析</summary>

中间结果贵，不只是因为它存在，而是因为它往往要被写回内存、再被下一步算子读回来。

如果一个计算图拆成很多小算子，每个算子都产生临时张量，那么这些张量就会频繁地在计算单元和内存之间往返。这样一来，真正耗时的可能不是计算，而是 memory traffic。

所以算子融合的第一层价值，是减少中间张量的落地次数。少一次写回、少一次读取，通常就意味着更少的带宽压力和更高的吞吐。
</details>
### Q1小验证：中间张量为什么拖慢系统

先记住“写回 + 再读回”这件事本身就很贵。

```python
def memory_traffic(num_ops, tensor_mb):
    # 每个算子都会把临时结果写回并再次读入。
    writes = max(num_ops - 1, 0)
    reads = max(num_ops - 1, 0)
    return (writes + reads) * tensor_mb

unfused = memory_traffic(4, 64)
fused = memory_traffic(1, 64)
print('unfused traffic MB:', unfused)
print('fused traffic MB:', fused)
print('reduction factor:', unfused // max(fused, 1))

```

## Q2：算子融合为什么能改善数据局部性？

<details>
<summary>点击展开查看解析</summary>

融合把原本分开的多个步骤合成一条更连续的执行链。

这样做的结果是：
- 中间结果更容易留在片上；
- 同一批数据更容易被连续复用；
- 数据搬运和计算之间的间隔更短。

数据局部性提升后，硬件就更容易把已有数据重复利用起来，而不是每一步都重新去 HBM 找一遍。这就是为什么融合不只是“少几个函数”，而是直接改变了数据流路径。
</details>
### Q2小验证：局部性为什么和融合有关

把“连续使用同一批数据”这件事记牢，就能理解融合的核心收益。

```python
def locality_score(reuse_steps, live_range):
    # 同一批数据被连续复用得越久、在片上停留越久，局部性越好。
    return reuse_steps * live_range

plans = {
    'separate': locality_score(1, 1),
    'partially_fused': locality_score(2, 2),
    'fully_fused': locality_score(3, 3),
}
for name, score in plans.items():
    print(name, '->', score)
print('best:', max(plans, key=plans.get))

```

## Q3：为什么融合和编译器、kernel 实现会绑定在一起？

<details>
<summary>点击展开查看解析</summary>

融合不是简单地把两个函数手动拼起来，而是需要在编译和实现层上重新安排计算顺序、临时变量和内存访问。

编译器能帮助决定哪些操作可以合并，kernel 实现则决定合并后如何在硬件上真正落地。比如：
- 哪些中间值可以直接在寄存器或 shared memory 中使用；
- 哪些步骤可以不必回到 HBM；
- 哪些访存模式可以一起优化。

所以融合是一个“编译决策 + kernel 实现”的组合问题。理解这一点后，后面看 Triton、CUDA 或 AI compiler 里的 fusion 逻辑就不会把它当成纯术语。
</details>
### Q3小验证：融合到底要改什么

先想内存路径，再想代码结构。

```python
def fusion_decision(compiler_can_fuse, kernel_can_hold, needs_reorder):
    # 三个条件都满足时，融合才更可能真正落地。
    return compiler_can_fuse and kernel_can_hold and needs_reorder

cases = [
    ('easy', True, True, True),
    ('compiler_only', True, False, True),
    ('kernel_limit', True, True, False),
]
for name, c, k, r in cases:
    print(name, '->', 'fuse' if fusion_decision(c, k, r) else 'defer')

```

## ⚠️ 常见误区

- 算子融合不是让代码变长，而是让中间结果少落地。
- 融合的目标不是“看上去更高级”，而是减少 memory traffic。
- 编译器和 kernel 都会参与融合，不能只看一侧。
- 先看数据流，再看代码名，通常更容易判断是否真的值得融合。