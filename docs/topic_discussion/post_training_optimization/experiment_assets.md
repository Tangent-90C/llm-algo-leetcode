# 后训练优化实验资产表

本表登记后训练机制页和项目页使用的环境、工作负载、产物与证据等级。环境创建见[实验 Quickstart](./quickstart.md)，项目公共结果字段见[Part 02 结果资产与证据契约](../../part02_result_assets.md)。

## 环境角色

| 环境 ID | 适用入口 | 可形成的证据 |
|:---|:---|:---|
| `PT-CPU` | 13–18、30–33、50–51 与三页 PT 机制页的 Step1–4 | labels、loss、adapter、偏好、reward、freshness 与决策机制 |
| `PT-SFT-CUDA` | 14、15、30、31；62–64 | 真实 LoRA/SFT 更新、长上下文训练代价、LoRA 变体路径、验证集与训练产物 |
| `Q-QLORA` | 26、65 | NF4 QLoRA、adapter 与低资源训练证据 |
| `PT-RL-CUDA` | 17、18、PT-REWARD-SIGNAL；84、85 的单卡阶段 | DPO 成对偏好边际、GRPO 组内优势与奖励信号的真实模型配对实验；17、18 已完成 Step 5，待真实 GPU 复测 |
| 项目指定多 worker 环境 | 85、86 | rollout 队列、策略同步、通信、恢复和扩展证据 |

## 机制页职责与学习级别

机制页不构成一条必须全部运行的线性路径。共同基础建立可复用语义，主路径形成最小闭环，分支页只在对应问题出现时进入。

| 页面 | 职责 | 学习级别 | GPU 状态 |
|:---|:---|:---|:---|
| [Part 02 · 33 微调准入](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md) | 根据失败证据选择 prompt、RAG、SFT 或偏好优化 | 共同入口 | 不需要 |
| [Part 02 · 32 SFT 数据准入](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) | 检查样本身份、去重、去污染和长度分布 | SFT 主路径 | 不需要 |
| [Part 02 · 13 SFT 训练循环](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) | 把 chat template、labels 与有效监督 Token 接入 loss | SFT 主路径 | 不需要；由 Part 02 · 15 承接真实训练 |
| [Part 02 · 14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.md) | 建立 target modules、rank、alpha 与 adapter 生命周期 | SFT 主路径 | 已有 `PT-SFT-CUDA` |
| [Part 02 · 15 端到端微调](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.md) | 验证训练、validation、checkpoint 与交付闭环 | SFT 主路径 | 已有 `PT-SFT-CUDA` |
| [Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md) | 解释 NF4、冻结基座与低资源 adapter 更新 | 低资源分支 | 复用 `Q-QLORA` |
| [Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md) | 比较 packing、序列长度和显存预算 | 长上下文分支 | 已有 `PT-SFT-CUDA` |
| [Part 02 · 31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md) | 比较 rsLoRA、DoRA、rank 与固定参数预算 | 参数适配分支 | 已有 `PT-SFT-CUDA` |
| [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) | 固定偏好对、数据隔离和独立评测协议 | 偏好共同基础 | 不需要 |
| [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md) | 解释 Reference、beta 与成对偏好更新 | 离线偏好分支 | 已建 `PT-RL-CUDA` Step 5；待实机验证 |
| [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) | 统一奖励、回报、价值与优势的关系 | 奖励优化共同基础 | 复用 `PT-RL-CUDA`；待补 Step 5 |
| [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md) | 解释 actor、reference、reward、critic 与资源状态 | PPO 分支 | 不增加机制级单卡实验 |
| [Part 02 · 18 GRPO](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md) | 解释候选组、组内基线、奖励归一化与 RLVR | GRPO / RLVR 分支 | 已建 `PT-RL-CUDA` Step 5；待实机验证 |
| [Part 02 · PT-ROLLOUT-LIFECYCLE Rollout 与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) | 管理轨迹身份、策略版本、freshness 与准入 | 在线系统共同基础 | 不需要 |
| [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) / [Part 02 · 51 Online DPO](../../02_PyTorch_Algorithms/51_Online_DPO.md) | 分别承担多阶段系统编排与在线偏好反馈 | 两条系统分支 | 留给多 worker 或项目验证 |

## 机制与项目接口

| 机制链 | 机制页产出 | 项目页补充 |
|:---|:---|:---|
| SFT 数据 → labels → loss | 13、32 | 64 的数据准入与 62 的真实训练交付 |
| LoRA / QLoRA → adapter | 14、26、31 | 31 输出固定 rank / 固定预算机制证据；63/65 承接质量、资源与 artifact gate |
| 偏好对 → DPO | 17、50 | 84 的训练—评测—决策记录 |
| reward/value/advantage → PPO 或 GRPO | Reward 页为共同基础，16 与 18 为并列算法分支 | 85 的组采样、奖励与更新闭环 |
| policy freshness → RL 系统或在线偏好 | Rollout 生命周期页为共同基础，RL 系统页与 51 为并列系统分支 | 85/86 的版本、编排、回滚与在线成本证据 |

## PT-RL-CUDA 机制工作负载

三项机制实验共享模型快照和单卡环境；17、18 共享 LoRA 有限更新路径，但各自保留候选变量，奖励信号页只比较固定轨迹上的信号构造。固定小样本用于隔离机制差异，不能替代独立偏好评测或在线系统验证。

| 页面 / 工作负载 | 固定条件 | 受控候选 | 结果路径 | 当前状态 |
|:---|:---|:---|:---|:---|
| Part 02 · 17 / `pt-dpo-real-model-update` | Qwen2.5-0.5B-Instruct、4 组偏好对、LoRA、4 次更新 | `beta = 0.05 / 0.10 / 0.30` | `benchmarks/results/pt_dpo/` | 代码与契约已完成，待实机验证 |
| Part 02 · 18 / `pt-grpo-real-model-update` | Qwen2.5-0.5B-Instruct、2 组固定候选、规则奖励、LoRA、4 次更新 | `group_size = 2 / 4` | `benchmarks/results/pt_grpo/` | 代码与契约已完成，待实机验证 |
| Part 02 · PT-REWARD-SIGNAL / `pt-reward-signal-matrix` | Qwen2.5-0.5B-Instruct、固定提示与回复、固定奖励 | KL 系数、终止奖励放置与优势构造 | `benchmarks/results/pt_reward_signal/` | 待建设 Step 5 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、baseline/candidate |
| 模型与数据 | model/tokenizer revision、chat template、dataset revision、split 与样本 hash |
| 训练 | dtype、有效 batch、学习率、更新次数、随机种子与策略版本 |
| 质量与安全 | loss、任务指标、win-rate/judge 一致性、安全与回滚门槛 |
| 资源与产物 | 峰值显存、step/rollout 时间、artifact/adapter 路径与 hash |
| 结论 | `failure`、`evidence_level`、decision 与下一步 |
