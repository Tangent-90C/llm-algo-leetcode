# 部署问题链：从模型产物到可交付服务

部署工作应沿着同一条因果链推进：

`模型与 artifact → 输入契约 → backend 加载 → 设备兼容 gate → 数据路径 → 服务拓扑 → workload / 质量验证 → 发布与成本决策`

![部署与异构学习路线：从产物到部署决策](../../docs/public/topic_discussion/deployment_heterogeneous/deployment_heterogeneous_roadmap.svg)

先确认候选身份。模型 revision、tokenizer、template、artifact、backend 与关键配置共同定义一次运行；只记录模型名不足以复现。接着通过设备、artifact、runtime 与关键 kernel 的兼容 gate，再观察请求在 host、accelerator、缓存与网络之间如何流动。

单卡不能满足容量、并发或尾延迟时，才进入副本、多卡、PD 分离、端边云协同或异构路由。每增加一层服务拓扑，都要同时记录状态交接、通信等待和失败/降级路径。

最后在固定 workload 和质量门槛下比较候选。性能数字必须附带硬件、backend、artifact、输入长度、并发、重复和 evidence level；否则只能作为线索，不能作为部署结论。

相关正文依次为 [01 部署契约](./01_model_artifact_and_deployment_contract.md) → [02 单机执行](./02_backend_compatibility_and_single_node_execution.md) → [07 兼容性](./07_accelerator_runtime_compatibility.md) → [03 数据路径](./03_device_coordination_and_data_path.md) → [04 服务分层](./04_multi_device_and_service_tiers.md) → [08 端边云放置](./08_edge_cloud_placement_and_handoff.md) → [05 回归发布](./05_observability_regression_and_release.md) → [06 选型决策](./06_heterogeneous_selection_and_cost_decision.md)。
