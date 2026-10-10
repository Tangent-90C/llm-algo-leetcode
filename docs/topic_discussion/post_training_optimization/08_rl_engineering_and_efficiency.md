# 08 · RL 工程、Rollout 与效率

RL 训练的成本来自生成系统与训练系统的协同，而不是单一损失。一次可重复的闭环需要把轨迹采样、奖励或验证器、参数更新、评测和恢复分成可观察阶段；否则 GPU 空闲、队列等待、通信暴露或资源竞争会被混成“训练很慢”。

进入框架选型前，先用 [Rollout 与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) 固定轨迹身份、策略版本和更新准入，再用 [采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) 比较共置、解耦与混合拓扑。框架负责执行这些契约，不能替代契约本身。

## 从生成到更新的工程闭环

| 阶段 | 输入 → 输出 | 主要资源问题 | 最小证据 |
| --- | --- | --- | --- |
| 轨迹采样 | 提示 → 多个回复 / 轨迹 | 执行后端吞吐、KV Cache、长度波动、队列等待 | 输入 / 输出 Token、TTFT、TPOT、失败原因 |
| 奖励与验证 | 轨迹 → 分数 / 终止信号 | 打分并发、超时、规则版本与不可判定样本 | 奖励组成、验证器版本、失败类型 |
| 参数更新 | 轨迹 + 奖励 → 候选策略 | 激活、优化器、参考策略、通信与恢复 | 单步时间、显存、检查点、重试记录 |
| 独立评测 | 基线 / 候选 → 决策 | 数据隔离、质量回归、安全与成本 | 质量门槛、失败样本、决策与回滚点 |

![RL 工程闭环：轨迹采样、奖励、更新、评测与恢复](../../public/topic_discussion/post_training_optimization/rl_engineering_loop.svg)

先测阶段边界，再解释总耗时。轨迹采样是推理工作负载，参数更新是训练工作负载；二者即使使用同一模型，也不能共享同一份吞吐或显存结论。工作进程扩大后，还需要分别记录等待、通信和恢复，而不是只报告一个端到端时间。

## 框架选择服务于实验阶段

| 条件 | 首选工具 / 组合 | 选择理由 | 仍需记录 |
| --- | --- | --- |
| 验证奖励、组采样或小规模更新语义 | TRL | 路径短，便于检查算法输入输出 | 模型、参考策略、奖励、数据与质量结果 |
| 需要独立采样后端 | vLLM 或 SGLang + 训练框架 | 将生成阶段与更新阶段分别观察 | 执行后端、长度、并发、队列与失败 |
| 需要多工作进程的 RL 系统 | verl 或 OpenRLHF | 提供采样、训练和资源编排的系统入口 | 工作进程拓扑、通信、恢复与资源占用 |
| 需要跨资源调度 | Ray 或框架内置调度 | 显式处理资源分配、重试与阶段并发 | GPU 分配、等待、超时和回滚点 |

框架名称不构成算法或性能结论。比较前固定模型、候选数、最大生成长度、奖励函数、并行规模、评测集和版本；先用最小闭环确认机制，再按证据逐步扩大系统。

## 验证与出口

[85 GRPO 组内对齐项目](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.md) 验证组采样和奖励闭环，[86 在线 DPO 基准](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.md) 比较更新收益、稳定性与回滚条件。真实资源瓶颈回补[性能优化](../performance_optimization/intro.md)，多工作进程通信回补[通信与并行](../communication_parallelism/intro.md)。相关阅读：[verl 文档](https://verl.readthedocs.io/en/latest/)、[OpenRLHF 文档](https://openrlhf.readthedocs.io/en/latest/)。
