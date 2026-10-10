# 05. 成本模型与 Profiling

## 页面目标

本页回答两个问题：如何从 shape、dtype 和资源约束生成候选配置，以及如何用 profiling 判断瓶颈是否真的被改善。Autotune 是搜索过程，profiling 是证据过程，二者不能互相替代。

![Autotune 与 Profiling：从候选配置到条件化结论](../../public/topic_discussion/operator_optimization/operator_autotune_evidence.svg)

## 成本模型的输入

| 输入 | 影响的决策 | 需要保留的记录 |
|:---|:---|:---|
| shape / dtype | tile、向量化和 Tensor Core 路径 | 输入分布、对齐方式 |
| 数据复用 | shared memory、register 和 fusion | 读写次数、工作集 |
| GPU 资源 | block、occupancy 和并发 | GPU 型号、编译目标 |
| workload | 候选配置和最终结论 | batch、序列长度、重复次数 |

## 为什么最优解会随 backend 分化

同一张图可以有多个语义等价的实现，但不同 backend 对 layout、Tensor Core 路径、编译缓存、launch 和 runtime 调度的成本假设不同。成本模型的目标不是宣布某个 backend “更快”，而是解释当前硬件与 workload 下为什么某个候选更合适。

| 分化来源 | 需要固定的条件 | 不能省略的记录 |
|:---|:---|:---|
| 硬件资源 | GPU 架构、显存与 SM 资源 | 设备型号、驱动、编译目标 |
| backend 能力 | 算子覆盖、图捕获、fallback | backend 版本、实际执行路径 |
| 输入分布 | shape、dtype、layout 与动态范围 | bucket 规则、输入分布 |
| 编译与缓存 | 首次编译、缓存命中与重编译 | compile time、cache key、冷/热运行 |

动态 Shape 需要单独记录。一个配置在固定 shape 上最优，不代表它适合其他 batch、序列长度或隐藏维度；autotune 还可能把编译时间、缓存命中和运行时间混在一起。

| 动态因素 | 需要比较 | 建议记录 |
|:---|:---|:---|
| Shape | 固定配置、shape bucket、按 shape 搜索 | shape 分布、bucket 规则 |
| 配置缓存 | 首次编译与缓存命中后运行 | compile time、cache key |
| dtype | 不同输入 / 累加路径 | 输入 dtype、累加 dtype |
| workload | 小 batch、长序列、混合请求 | 各组独立结果，不只报平均值 |

## Profiling 证据链

先提出瓶颈假设，再用相同 workload 采集 kernel 时间、访存、occupancy、编译成本和端到端指标。一次 trace 只能描述当前配置；要形成结论，需要 baseline、候选、重复运行和失败条件。

| 现象 | 可能原因 | 下一步检查 |
|:---|:---|:---|
| kernel 快，端到端不快 | launch、同步或其他阶段占主导 | trace 与调用占比 |
| memory throughput 低 | 布局、复用或 tile 不合适 | load/store、cache、stride |
| occupancy 下降 | register / shared memory 压力 | spill、block 资源 |
| 不同 shape 波动大 | 配置泛化不足 | shape 分桶、配置缓存和 autotune |

## 本页出口

你应能把一个 profiling 现象映射到可验证的优化动作，并说明为什么“某个 shape 上最快”不等于“所有 workload 上最优”。

统一记录至少包括：shape、dtype、block 配置、num warps、num stages、backend、显存上限、延迟上限、编译时间、重复运行结果和最终选择理由。算子优化路线中的 Task 5 负责成本模型与 Profiling，Task 6 把候选搜索接入端到端项目。
