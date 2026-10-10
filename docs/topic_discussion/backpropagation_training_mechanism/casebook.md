# 反向传播与训练机制问题索引

当训练出现梯度为空、更新不一致、数值异常或显存峰值时，先按现象定位机制，再进入对应正文与 Notebook 验证。

## 按现象回查

| 现象 | 优先检查 | 核心问题 | 入口 |
|:---|:---|:---|:---|
| 损失无法连接到参数 | 训练闭环与计算图 | 前向传播、损失和待训练参数是否处于同一计算图？ | [01](./01_training_loop_and_computation_graph.md) |
| 参数 `.grad` 为 `None` | 自动求导与梯度边界 | 参数是否冻结，计算图是否被 `detach()` 或 `no_grad()` 切断？ | [02](./02_autograd_and_transformer_backward.md) |
| 非标量输出无法执行反向传播 | VJP | 是否提供了形状正确的上游梯度？ | [02](./02_autograd_and_transformer_backward.md) |
| 残差或共享参数梯度异常 | 分支梯度汇合 | 多条路径的梯度贡献是否正确相加？ | [01](./01_training_loop_and_computation_graph.md)、[02](./02_autograd_and_transformer_backward.md) |
| 损失有数值但监督效果不可信 | 监督边界 | 标签移位、损失掩码、`ignore_index` 与有效 token 是否一致？ | [03](./03_supervision_gradient_scale_effective_update.md) |
| 梯度累积与完整批次不一致 | 梯度尺度 | 损失归约、分母与累计窗口是否对齐？ | [03](./03_supervision_gradient_scale_effective_update.md) |
| 学习率调度器推进过快 | 更新节奏 | 学习率调度器是否跟随优化器更新步？ | [03](./03_supervision_gradient_scale_effective_update.md) |
| 理论账本与峰值对不上 | 状态生命周期 | 哪些状态在峰值阶段仍同时驻留？ | [04](./04_training_state_lifecycle.md) |
| 激活检查点后输出或梯度变化 | 重算正确性 | RNG、参数、输入与外部状态是否可重放？ | [05](./05_checkpointing_and_offload.md) |
| 卸载后等待时间过长 | 搬运关键路径 | 搬出、搬回和反向传播消费点是否重叠？ | [05](./05_checkpointing_and_offload.md) |
| FP16 出现 NaN/Inf | 梯度安全 | 损失缩放、梯度尺度还原、溢出检查和梯度裁剪顺序是否正确？ | [06](./06_correctness_and_performance_evidence.md) |
| 显存下降但训练更慢 | 联合性能证据 | 容量收益是否被重算或传输时间抵消？ | [06](./06_correctness_and_performance_evidence.md) |

## 按证据选择下一步

| 已有证据 | 下一项检查 |
|:---|:---|
| 只有损失曲线 | 增加监督计数、梯度有限值和更新次数 |
| 只有理论显存账本 | 增加阶段峰值与 allocator 统计 |
| 只有 GPU 峰值 | 回查主导状态及其生命周期 |
| 只有性能分析时间线 | 先确认基线和候选的正确性与工作负载一致 |
| 只有一次运行结果 | 增加预热、重复测量、失败记录和环境信息 |

## 跨专题出口

- 训练目标、数据和 SFT/DPO/GRPO 项目：进入[后训练优化](../post_training_optimization/intro.md)。
- 峰值显存、时间线追踪、吞吐和策略组合：进入[性能优化](../performance_optimization/intro.md)。
- 反向传播内核、融合和 I/O：进入[算子优化](../operator_optimization/intro.md)。
- 梯度同步、状态切分和多卡扩展：进入[通信与并行](../communication_parallelism/intro.md)。
