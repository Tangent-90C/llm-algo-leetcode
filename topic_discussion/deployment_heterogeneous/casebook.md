# 部署与异构决策手册

| 现象 | 先检查什么 | 可能进入的正文 | 不应直接得出的结论 |
|:---|:---|:---|:---|
| 模型或量化产物无法加载 | artifact 格式、revision、tokenizer、backend 版本 | [01](./01_model_artifact_and_deployment_contract.md)、[02](./02_backend_compatibility_and_single_node_execution.md) | 文件存在就代表 backend 支持 |
| 输出与预期不一致 | template、special token、采样与停止条件 | [01](./01_model_artifact_and_deployment_contract.md)、[02](./02_backend_compatibility_and_single_node_execution.md) | 延迟可比就代表质量可比 |
| 模型能加载但在目标设备上报错或明显变慢 | 实际 dtype、设备能力、量化 kernel、runtime 插件与 fallback | [07](./07_accelerator_runtime_compatibility.md) | 能启动就代表目标执行路径可用 |
| GPU 利用率低或端到端慢 | CPU 准备、host-device copy、缓存、同步和网络 | [03](./03_device_coordination_and_data_path.md) | GPU 利用率低一定是 GPU 性能不足 |
| 单机容量不足或尾延迟失控 | 权重/KV 容量、并发、通信、handoff 与路由 | [04](./04_multi_device_and_service_tiers.md) | 加卡一定线性扩展 |
| 数据边界、网络等待或区域容量限制改变服务设计 | 输入/权重/KV/输出的放置、RTT、handoff 与降级 | [08](./08_edge_cloud_placement_and_handoff.md) | 端、边、云层级越多体验一定越好 |
| 更新后性能或质量回归 | workload、环境、artifact、执行路径与错误样例 | [05](./05_observability_regression_and_release.md) | 单次 benchmark 代表版本稳定 |
| 多候选难以取舍 | 质量、容量、SLO、成本、维护与回滚 | [06](./06_heterogeneous_selection_and_cost_decision.md) | 吞吐最高就是最佳部署方案 |

这份手册用于从症状跳到适合的机制与证据页，不替代 Task0–6 的学习顺序。
