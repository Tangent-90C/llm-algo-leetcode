# 性能优化判断手册

本手册按现象选择测量对象与下一步验证，不替代 Task0–6 的连续学习。每次都先回到固定 workload、baseline 与质量门槛，避免把不同运行条件下的结果直接比较。

| 现象 | 优先检查 | 先进入 | 不能直接推出的结论 |
|:---|:---|:---|:---|
| step time 变长 | forward、backward、optimizer、数据加载与同步 | [性能分析](../performance_analysis/intro.md) → [02 归因证据](./02_observability_and_profiling.md) → [03 训练账本](./03_training_memory_ledger.md) | 不能只凭一个慢 kernel 定位原因 |
| 峰值显存升高 | 参数、activation、optimizer、cache、workspace 与通信缓冲区 | [03 训练账本](./03_training_memory_ledger.md) 或 [04 推理状态容量](./04_inference_memory_and_kv_cache.md) | 显存下降不等于端到端更快 |
| 有效 batch 不变但单步峰值或 step time 改变 | micro-batch、accumulation、activation 与更新次数 | [03 训练账本](./03_training_memory_ledger.md) | effective batch 不等于单次 forward 的显存口径 |
| DPO / GRPO 训练慢但 GPU 并未持续繁忙 | policy/reference 驻留、响应长度、rollout、reward/verifier 与 update 比例 | [03 后训练资源账本](./03_training_memory_ledger.md) → [05 关键路径](./05_compute_communication_and_io.md) | loss 或 reward 变化不能说明系统瓶颈 |
| 理论账本与 GPU 实测差距很大 | allocator 保留、workspace、碎片、通信 buffer 与生命周期重叠 | [性能分析](../performance_analysis/intro.md) → [02 归因证据](./02_observability_and_profiling.md) → [03 训练账本](./03_training_memory_ledger.md) | 理论字节数不能替代峰值和 reserved 的实测 |
| 模型加载、权重转换或首请求特别慢 | artifact、磁盘/网络、H2D、初始化与 warmup | [性能分析](../performance_analysis/intro.md) → [02 归因证据](./02_observability_and_profiling.md) → [05 关键路径](./05_compute_communication_and_io.md) | 首次加载慢不等于 steady-state 吞吐差 |
| GPU 利用率低 | CPU 等待、I/O、同步、通信和 kernel 空洞 | [05 关键路径](./05_compute_communication_and_io.md) | 利用率低不一定是算力不足 |
| profiler 内耗时明显变差 | 采集范围、同步、trace 开销与非 profiler 复测 | [性能分析](../performance_analysis/intro.md) | profiler 内绝对耗时不能直接当作真实性能 |
| 吞吐提高但延迟变差 | batch、队列、P99、TTFT / TPOT 与并发 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md) → [06 成本与准入](./06_cost_model_and_production_decision.md) | 平均吞吐不能代表交互质量 |
| KV handoff 或网络等待拉长 | transfer bytes、序列化、网络、分池和等待 | [05 关键路径](./05_compute_communication_and_io.md) | 状态已迁移不等于 handoff 被隐藏 |
| rollout 生成快但有效更新吞吐低 | 无效响应、verifier、环境失败、权重刷新和策略新鲜度 | [05 关键路径](./05_compute_communication_and_io.md) → [06 成本与准入](./06_cost_model_and_production_decision.md) | 原始生成 tokens/s 不能代表后训练有效吞吐 |
| 流式输出、日志或 checkpoint 写入拖慢尾部 | 回压、磁盘/网络写入、CPU 编码与同步点 | [05 关键路径](./05_compute_communication_and_io.md) | 输出慢不一定来自 decode kernel |
| 量化后显存下降但质量或 P99 退化 | dtype、量化粒度、artifact、backend 路径与质量集 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md) → [06 成本与准入](./06_cost_model_and_production_decision.md) | 低比特账本收益不等于部署收益 |
| 多卡扩展不理想 | collective、拓扑、通信占比、负载与重叠 | [05 关键路径](./05_compute_communication_and_io.md) | GPU 数量增加不保证线性加速 |
| 局部策略有效但组合后退化 | 组合改变的容量、关键路径、质量和回退条件 | [07 端到端决策](./07_end_to_end_performance_engineering.md) | 单项收益不能直接相加 |

最小记录模板：`现象 → workload / baseline → 阶段与对象 → 证据等级 → candidate 及代价 → 非 profiler 复测 → 质量与成本 → accept / tune / reject`。
