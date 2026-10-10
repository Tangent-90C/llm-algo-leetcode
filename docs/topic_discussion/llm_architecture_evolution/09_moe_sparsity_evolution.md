# 09 MoE 与稀疏容量（MoE and Sparse Capacity）

## 学习问题

混合专家（Mixture of Experts，MoE）把模型总容量与每个词元的激活计算拆开：路由器为词元选择少量专家，专家完成 FFN 变换，再将结果合并回主干。判断收益时需要同时检查结构容量、训练稳定性和通信代价。

## 核心机制

### 设计压力：容量增长为什么不应等于每个词元的成本增长？

稠密 FFN 让所有词元使用同一组参数，模型越宽，每个词元的成本越高。MoE 通过稀疏激活增加总参数，但能否获得收益取决于路由质量、专家容量和系统执行是否匹配。

![MoE 稀疏容量：路由器选择专家，分发后再合并](../../public/topic_discussion/llm_architecture_evolution/moe_sparsity.svg)

### 结构机制：选择、容量与回收缺一不可

对含 $T$ 个词元、$E$ 个专家和 Top-$k$ 路由的批次，平均每个专家接收 $Tk/E$ 个词元。若容量因子为 $c$，单专家容量通常按 $\lceil cTk/E\rceil$ 设定。$c$ 太小会产生溢出，太大则预留大量空槽；专家数或 Top-$k$ 改变时，容量、通信负载和激活计算必须一起重算。

共享专家保证每个词元都有公共路径，细粒度专家改变单个专家的容量和组合空间，无辅助损失路由则尝试减少均衡目标对主任务的干扰。三者改变的对象不同，仍需用专家计数、容量溢出率、分发字节和质量指标分别验证。

| 环节 | 机制问题 | 失败表现 |
|:---|:---|:---|
| 路由器 / Top-k | 哪些专家接收当前词元 | 路由塌缩、专家长期闲置 |
| 专家容量 / 容量溢出 | 单个专家最多接收多少词元 | 词元丢弃、回退或排队 |
| 分发 / 合并 | 词元如何发送并回收结果 | All-to-All 暴露、重排开销 |
| 负载均衡目标 | 如何约束专家利用率 | 辅助损失干扰主任务或约束不足 |

## 选择条件与验证证据

评估 MoE 至少记录 Top-k、专家数、容量因子、容量溢出、专家负载分布、分发字节数、通信暴露时间与任务质量。仅报告总参数量或理论激活参数，无法判断真实吞吐和稳定性。

- [Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md)：区分总参数与激活计算。
- [Part 02 · 06 MoE 路由](../../02_PyTorch_Algorithms/06_MoE_Router.md)：实现 Top-k 与容量约束。
- [Part 02 · 07 MoE 负载均衡](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md)：分析专家分布与稳定性。
- [Part 02 · 46 MoE 专家并行](../../02_PyTorch_Algorithms/46_MoE_Expert_Parallel.md)：连接分发、合并与通信。
- [Part 02 · 80 MoE 专家并行项目](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md)：形成真实多卡证据。

## 相关阅读

- [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
