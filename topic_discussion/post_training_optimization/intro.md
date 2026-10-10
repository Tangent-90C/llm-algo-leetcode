# 后训练优化（Post-Training Optimization）

> 专题类型：主学习路线　主服务目标：从监督微调基线、偏好优化到强化学习与采用决策

后训练优化研究如何在基础模型之上，利用监督数据、偏好反馈或可验证奖励改变模型行为。学习者需要判断问题是否值得通过参数更新解决，在监督微调基线之上选择离线偏好优化或在线策略优化，并依据独立评测、资源成本和回滚条件决定是否采用候选模型。

![完整训练生命周期中的后训练位置](../../docs/public/topic_discussion/post_training_optimization/training_lifecycle_context.svg)

## 学习路线

按 Task 顺序学习时使用下表；需要连续理解因果关系时进入[后训练优化问题链](./walkthrough.md)，已有数据、训练日志或评测报告时使用[后训练优化判断手册](./casebook.md)，术语差异查询[专题术语表](./glossary.md)。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 目标诊断与阶段准入 | 当前失败能否由 Prompt、RAG 或工具解决；若必须更新参数，应停在 SFT、进入离线偏好，还是需要在线奖励与探索？ | 核心：[后训练问题链 · 阶段选择](./walkthrough.md#一先判断是否需要改变模型参数) → [Part 02 · 33 微调准入与最小干预](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb) | 稳定失败证据 → 知识 / 行为 / 偏好 / 探索根因 → 最小干预 → 数据、反馈与评测准入 → 训练分支 | [04 SFT 完成与阶段升级](./04_end_to_end_experiment.md#sft-何时完成何时升级) |
| Task1 · 监督目标与最小 SFT 基线 | 样本怎样经过切分、模板和标签掩码变成有效监督 Token，并完成一次可复查的 SFT 基线？ | 核心：[Part 02 · 32 SFT 数据准入与长度审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb) → [Part 02 · 13 SFT 训练循环](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.ipynb) → [Part 02 · 15 端到端微调](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.ipynb)；基础回补：[Part 02 · 11–12 训练控制](../../02_PyTorch_Algorithms/2_3.md)；项目链（待重构）：[Part 02 · 64 数据准入 → 62 训练交付](../../02_PyTorch_Algorithms/2_10.md) | 来源、切分与去污染 → 模板、分词与标签边界 → 有效监督词元与损失 → 有效更新 → 冻结验证 → 可复查基线 | [01 SFT 数据与监督契约](./01_sft_data_and_loss.md) → [04 SFT 基线与交付](./04_end_to_end_experiment.md) |
| Task2 · 更新范围与参数高效适配 | 训练目标不变时，哪些参数需要更新，怎样在全参数、LoRA、QLoRA 和长上下文方案之间选择？ | 核心：[Part 02 · 14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.ipynb)；扩展：[Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) / [30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb) / [31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb)；项目链（待重构）：[Part 02 · 63 策略选择 → 62 训练交付](../../02_PyTorch_Algorithms/2_10.md)；交叉项目：[Part 02 · 65 QLoRA 选择](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb) | 全参 / 适配器选择 → 目标模块、秩与缩放 → QLoRA / 长上下文 / LoRA 变体分支 → 固定口径比较 → 模型或适配器产物 | [02 LoRA / PEFT](./02_lora_peft_design.md) → [03 训练控制](./03_training_control.md) → [04 产物与交付](./04_end_to_end_experiment.md#最小实验闭环) |
| Task3 · 偏好数据与离线策略优化 | 已有同提示的优选与非优选回答时，怎样固定参考策略、形成偏好边际并完成离线 DPO 更新？ | 核心：[Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) → [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb)；项目（待重构）：[Part 02 · 84 DPO](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb) | 同提示偏好对 → 固定参考策略 → 偏好边际 → 离线更新与独立复测 | [05 DPO 与离线偏好优化](./05_dpo_and_preference_optimization.md) |
| Task4 · 奖励信号与在线策略更新 | 偏好对或验证结果怎样形成可信奖励；当前策略生成的回答又怎样经过基线和优势估计，形成 PPO / GRPO 更新？ | 核心：[Part 02 · PT-REWARD-MODEL 奖励模型训练、校准与过度优化](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb) → [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb)，再选择 [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb) 或 [Part 02 · 18 GRPO](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.ipynb)；项目（待重构）：[Part 02 · 85 GRPO / RLVR](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb) | 偏好或规则反馈 → 奖励训练与留出校准 → 在线采样 → 奖励、价值或组内基线 → 优势 → 受约束更新 → 隐藏质量审计 | [06 PPO / RLHF](./06_rlhf_and_ppo_system_cost.md) / [07 GRPO / RLVR](./07_grpo_and_groupwise_alignment.md) |
| Task5 · 轨迹生命周期与在线训练系统 | 多轮在线更新中，怎样判断轨迹仍可用于当前策略、同步新权重，并在队列积压、工作进程失败或恢复后继续训练？ | 核心：[Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb)，再选择 [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb) 或 [Part 02 · 51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.ipynb)；项目出口（待重构）：[Part 02 · 85–86](../../02_PyTorch_Algorithms/2_10.md) | 策略版本 → 轨迹身份与新鲜度 → 奖励状态 → 采样 / 奖励 / 训练拓扑 → 队列与更新交接 → 同步、检查点与恢复 | [08 RL 工程、轨迹采样与效率](./08_rl_engineering_and_efficiency.md) |
| Task6 · 独立评测与采用决策 | 训练后的候选与固定基线相比，是否同时通过目标能力、通用能力、安全、鲁棒性、成本和回滚门槛？ | 机制回补：[Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) / [Part 02 · 51 在线更新与回滚](../../02_PyTorch_Algorithms/51_Online_DPO.ipynb)；项目链（待重构）：[Part 02 · 84–86](../../02_PyTorch_Algorithms/2_10.md) | 冻结评测集与评分协议 → 基线 / 候选 → 目标增益与能力保持 → 安全、鲁棒性与评审一致性 → 资源、失败切片与回滚 → `accept / tune / reject` | [09 质量、在线反馈与项目决策](./09_preference_data_and_project_decision.md) |

![后训练路线：共同训练基础分流到偏好、奖励和在线系统](../../docs/public/topic_discussion/post_training_optimization/post_training_roadmap.svg)

![后训练知识地图：数据、信号、状态与证据如何连接](../../docs/public/topic_discussion/post_training_optimization/post_training_knowledge_map.svg)

## 扩展与跨专题入口

| 需要补齐的能力 | 首选入口 | 在本专题中的作用 |
|:---|:---|:---|
| 训练计算图、激活、检查点与卸载 | [反向传播与训练机制](../backpropagation_training_mechanism/intro.md) | 解释训练状态和策略代价 |
| 数据生命周期、清洗质量与后训练接口 | [数据生命周期与契约](../data_engineering/00_data_lifecycle_and_contract.md) / [清洗与质量链](../data_engineering/02_cleaning_quality_chain.md) / [后训练数据接口](../data_engineering/05_post_training_data_interfaces.md) | 为 SFT、偏好和轨迹数据提供来源、去污染、版本与交付证据 |
| 训练峰值、多模型驻留与轨迹采样吞吐 | [性能优化 Task2](../performance_optimization/03_training_memory_ledger.md) | 建立 SFT、QLoRA、DPO、PPO 和 GRPO 的资源账本 |
| QLoRA、低比特产物与部署格式 | [量化与压缩](../quantization_compression/intro.md) | 解释低比特训练底座与产物约束 |
| 多卡状态切分、轨迹通信与扩展 | [通信与并行](../communication_parallelism/intro.md) | 解释工作进程、通信与规模化代价 |
| 工具轨迹、环境反馈与沙盒执行 | [Agent 工作负载与轨迹](../agent/00_agent_workload_and_trace.md) / [轨迹数据](../agent/01_agent_trajectory_data.md) / [沙盒环境](../agent/02_agent_sandbox_environment.md) | 仅为工具调用与环境交互轨迹补充任务、回放和执行证据；普通文本轨迹仍由本专题管理 |

## 实验与项目出口

首次运行先按[实验 Quickstart](./quickstart.md)创建或恢复环境，再从[实验资产表](./experiment_assets.md)选择机制页、模型、候选和结果路径；跨专题的软件版本与硬件快照查询[GPU 环境资产表](../../docs/gpu_environment_assets.md)。

| 证据阶段 | 要回答的问题 | 输出与入口 |
|:---|:---|:---|
| CPU 机制 | 数据、目标、奖励和决策关系是否正确？ | 13–18、30–33、50–51 与四页 PT 机制页的 Step1–4 |
| 单卡监督训练 | 参数更新、长度和适配器路径是否可执行？ | `PT-SFT-CUDA`；14、15、30、31 |
| 单卡偏好与奖励 | 偏好边际、组内优势和奖励信号是否按预期变化？ | `PT-RL-CUDA`；17、18、PT-REWARD-SIGNAL |
| 多工作进程闭环 | 轨迹采样、奖励、更新、通信和恢复能否协同？ | 项目指定的分布式环境；状态登记见实验资产表 |
| 采用决策 | 候选是否通过质量、安全、资源与回滚门槛？ | 09 正文、50–51 机制页与后训练项目链 |
