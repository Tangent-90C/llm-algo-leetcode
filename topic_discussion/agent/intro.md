# Agent 基础设施（Agent Infrastructure）

> 专题类型：领域专题（建设中）　主服务目标：让模型从单次生成走向可观测、可评估、可复现的多步任务执行

## 页面导语

推理优化解决一次模型调用如何更快、更省；Agent Infra 处理的是多步任务如何可靠完成。一次 Agent 任务需要模型调用、工具调用、上下文管理、执行环境、轨迹日志与最终评估协同工作，workload 不再是固定的单轮请求，而是可能失败、回退和分支的动态调用图。

本专题复用推理优化的服务指标、后训练优化的数据与奖励、性能优化的 trace 证据，但聚焦工具使用、沙盒执行、轨迹调度与任务级评测这一条独立能力链。

## 如何开始

第一阶段从 Task0–2 开始：先建立组件全景，再学习轨迹数据生产和安全可复现的执行环境。已有 Serving 基础时再进入 Task3；已经具备训练和评测闭环后，才进入 Task5–6 的框架工程与 Agentic RL。

## Agent Infra Task0–6

| Task / 主题 | 要回答的问题 | 当前关联入口与后续建设 | 交付物 |
|:---|:---|:---|
| Task0 · Agent Infra 全景与核心链路 | 单次推理怎样演化为多步任务执行？ | [推理优化](../inference_optimization/intro.md) → [后训练优化](../post_training_optimization/intro.md) → [00 Agent 工作负载与执行链](./00_agent_workload_and_trace.md) | 模型、工具、上下文、沙盒、trace、评估的组件图与 workload 卡 |
| Task1 · Agent 数据生产与合成 | 如何生成包含工具调用和完整执行轨迹的训练数据？ | [数据工程](../data_engineering/intro.md) → [后训练优化 Task1/4](../post_training_optimization/intro.md) → [01 Agent 轨迹数据与验证](./01_agent_trajectory_data.md) | 任务蓝图、候选轨迹、环境反馈与验证规则 |
| Task2 · 沙盒与执行环境 | 如何隔离工具执行、限制资源并记录可复现反馈？ | [部署与异构](../deployment_heterogeneous/intro.md) → [02 Agent 沙盒与执行环境](./02_agent_sandbox_environment.md) | 工具 schema、资源配额、日志和环境快照 |
| Task3 · 轨迹感知调度与上下文管理 | 多阶段、可失败任务怎样调度和复用状态？ | [推理优化 Task4](../inference_optimization/intro.md) → [70 Serving 调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)；后续补 Agent 调度项目 | FCFS / 优先级 / 轨迹感知策略对照 |
| Task4 · 任务评估与可观测性 | 如何同时评估完成率、工具正确性、轨迹质量、延迟与成本？ | [性能优化](../performance_optimization/intro.md) → [50 评测协议](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb)；新建 Agent trace 评测页 | 结构化 trace、错误分类和任务级报告 |
| Task5 · 框架与 Harness 工程 | 编排、协作、编码 Agent 和 MCP 应怎样选型？ | 新建 LangGraph / AutoGen / MCP 选型页；连接[Part 05](../../05_Integrated_Projects_and_Decisions/intro.md) | 多工具调用 harness 与框架选型记录 |
| Task6 · Agentic RL 与综合项目 | 如何把数据生产、沙盒、rollout、奖励和评估收束为训练闭环？ | [后训练优化 Task4–6](../post_training_optimization/intro.md)；后续 Part 05 Agent 综合项目 | 可复现实验、任务质量/成本报告与决策 |

## 与现有专题的协作关系

| 专题 | Agent 复用什么 | Agent 专题新增什么 |
|:---|:---|:---|
| 推理优化 | 请求指标、调度、KV 状态与 backend | 动态轨迹、工具依赖、任务失败与重试 |
| 后训练优化 | SFT / 偏好数据、RL / RLVR、奖励与评测 | 工具轨迹、环境反馈与 Agentic RL 任务 |
| 数据工程 | 采集、清洗、去重、合成与血缘 | 任务蓝图、轨迹验证与执行反馈数据 |
| 性能优化 | trace、成本、资源与 benchmark 方法 | 任务级 SLO、工具等待、沙盒资源与端到端成功率 |

## 分阶段建设

Task0–2 已提供概念、数据与环境契约；下一阶段建设 Task3–4：轨迹感知调度与可观测性，并复用推理/性能的指标框架。Task5–6 留到前两阶段有固定 workload、工具 schema、沙盒与评测协议后再建设；在此之前不宣称任何 Agent 框架、模型或调度器具有通用优势。
