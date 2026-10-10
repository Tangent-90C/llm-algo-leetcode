# 03. 图级优化、融合与 Kernel 组合

## 页面目标

本页把图级候选收敛成 kernel 候选：先识别依赖、中间张量和可改写边界，再检查 lowering、layout 和调度是否保住预期收益。fusion 是其中最常见的一类图变换，而不是唯一目标。

![Fusion：减少写回，也重新分配执行资源](../../public/topic_discussion/operator_optimization/operator_fusion_lifecycle.svg)

## 从图结构筛选候选

图优化先回答“哪些变化值得进入执行路径”，而不是直接追求节点变少。常量折叠、死代码消除、公共子表达式消除、重排与 fusion 都必须保持依赖与数值契约；动态 shape、alias、随机性或跨设备同步会改变可行边界。

| 图级观察 | 可能的候选 | 先排除的风险 |
|:---|:---|:---|
| 连续生产—消费且结果只使用一次 | 融合或片上直传 | 中间值是否被其他分支读取 |
| 重复且输入相同的子图 | 公共子表达式复用 | dtype、shape 与随机状态是否一致 |
| 无消费者的节点 | 删除无效计算 | 是否承载副作用、日志或同步 |
| 可交换的独立节点 | 调整顺序或并行调度 | 是否引入额外 layout / copy |

## 融合、lowering 与调度

| 变化 | 可能收益 | 可能代价 |
|:---|:---|:---|
| 减少 kernel launch | 调度开销下降 | 单个 kernel 变复杂 |
| 减少中间张量写回 | HBM 流量下降 | register / shared memory 压力上升 |
| 增加片上复用 | 数据搬运减少 | tile 和 layout 约束变强 |
| 合并多个阶段 | 端到端路径变短 | 并行度、可调优空间可能下降 |

图级候选还需要经过三个连续检查：**legalization** 确认目标表示支持该语义；**lowering** 决定 tile、layout 与 kernel 组织；**schedule** 决定 launch、并行度和依赖顺序。能合法下沉不等于下沉后仍划算。

## 组合决策

先画出算子依赖和中间张量，再判断哪些阶段共享输入、哪些阶段必须保留同步点。Norm、激活、Softmax、QKV 或 Attention 的融合，应分别说明语义契约、复用关系和边界处理。节点减少只能说明图结构变化，不能直接说明性能收益。

| 检查项 | 需要回答的问题 |
|:---|:---|
| 正确性 | 融合前后的输出和误差是否对齐？ |
| 访存 | 哪些中间结果不再写回全局显存？ |
| 资源 | register、shared memory 和 occupancy 是否恶化？ |
| workload | 哪些 shape、dtype、batch 或序列长度真正受益？ |

| 下沉检查 | 需要回答的问题 | 常见失效原因 |
|:---|:---|:---|
| legalization | 目标 IR / backend 能表达该操作吗？ | 动态 shape、alias、数值语义不满足 |
| layout | 输入输出的布局能否直接复用？ | 为融合新增 transpose 或 copy |
| schedule | tile、并行度和 launch 是否保住图级收益？ | register 过高、occupancy 下降、同步增多 |
| fallback | 不支持时回到哪条路径？ | 静默回退使 benchmark 失去解释力 |

## 学习顺序

先阅读 Part 01 的算子融合与编译基础，再用 Part 03 的 RMSNorm、Softmax 和 FlashAttention 实现观察不同组合；需要更深的 IR 与 schedule 实现时，回查 Part 01 · 09、Part 04 · 18 与 Part 04 的 CUDA 实践，而不是切换到另一条专题路线。

## 本页出口

你应能为一次 fusion 提出可验证假设：减少了哪类搬运，增加了哪类资源压力，以及需要用哪些 kernel 和端到端指标验证。
