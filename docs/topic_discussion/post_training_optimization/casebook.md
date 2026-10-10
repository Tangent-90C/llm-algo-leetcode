# 后训练优化判断手册 | Post-Training Casebook

先根据问题所在的训练信号与工程证据选择下一步，而不是从某个算法名开始。

| 现象或目标 | 先检查什么 | 首选正文 | 项目出口 |
|:---|:---|:---|:---|
| 模型格式不稳、损失与监督对象不一致 | 聊天模板、标签、掩码、切分与数据质量 | [01 SFT 数据与监督契约](./01_sft_data_and_loss.md) | [Part 02 · 32 / 64](../../02_PyTorch_Algorithms/2_10.md) |
| adapter 无效、训练不可复现或有效 batch 不清 | target modules、rank、optimizer step、scheduler 与评测集 | [02 LoRA / PEFT](./02_lora_peft_design.md)、[03 训练控制](./03_training_control.md) | [62 / 63 / 65](../../02_PyTorch_Algorithms/2_10.md) |
| 训练完成但检查点、分词器或报告缺失 | 产物、恢复状态、样例输出与结果记录 | [04 SFT 实验与交付](./04_end_to_end_experiment.md) | [Part 02 · 62](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md) |
| 有偏好对，需要更轻的离线优化 | 优选 / 非优选回答质量、参考策略与独立评测 | [05 DPO](./05_dpo_and_preference_optimization.md) | [Part 02 · 84](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md) |
| 需要轨迹采样、奖励或组内比较 | 奖励、优势、候选组、KL 与验证器 | [06 PPO / RLHF](./06_rlhf_and_ppo_system_cost.md)、[07 GRPO / RLVR](./07_grpo_and_groupwise_alignment.md) | [Part 02 · 85](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.md) |
| 在线更新变慢、昂贵或不可回滚 | 采样后端、工作进程、策略新鲜度、资源与失败记录 | [08 RL 工程](./08_rl_engineering_and_efficiency.md) | [Part 02 · 86](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.md) |
| 指标相互冲突 | 质量、安全、鲁棒性、成本和评审器一致性 | [09 评测与项目决策](./09_preference_data_and_project_decision.md) | [Part 02 · 84–86](../../02_PyTorch_Algorithms/2_10.md) |

最小记录字段：`data / model / reference / configuration / workload / quality / resource / failure / artifact / decision`。

训练机制异常进入[反向传播与训练机制](../backpropagation_training_mechanism/intro.md)，真实资源瓶颈进入[性能优化](../performance_optimization/intro.md)，数据管道问题进入[数据工程](../data_engineering/intro.md)。
