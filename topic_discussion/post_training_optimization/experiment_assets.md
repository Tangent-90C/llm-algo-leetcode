# 后训练优化实验资产表

本表登记后训练机制页和项目页使用的环境、工作负载、产物与证据等级。环境创建见[实验 Quickstart](./quickstart.md)，项目公共结果字段见[Part 02 结果资产与证据契约](../../docs/part02_result_assets.md)。

## 环境角色

| 环境 ID | 适用入口 | 可形成的证据 |
|:---|:---|:---|
| `PT-CPU` | 13–18、30–33、50–51 与四页 PT 机制页的 Step1–4 | 标签、损失、适配器、偏好、奖励、策略新鲜度与决策机制 |
| `PT-SFT-CUDA` | 14、15、30、31；62–64 | 真实 LoRA/SFT 更新、长上下文训练代价、LoRA 变体路径、验证集与训练产物 |
| `Q-QLORA` | 26、65 | NF4 QLoRA、适配器与低资源训练证据 |
| `PT-RL-CUDA` | 17、18、PT-REWARD-SIGNAL；84、85 的单卡阶段 | DPO、GRPO 与奖励信号页均已完成 RTX 5070 Ti 实测 |
| 项目指定多工作进程环境 | 85、86 | 轨迹队列、策略同步、通信、恢复和扩展证据 |

## 单卡 GPU 验证顺序

两个 Profile 分开创建和验证，不在同一环境追加依赖。每页先使用默认工作负载生成独立 JSON；前一页失败时保留失败记录并修复，不跳过后汇总为专题结论。

验证前只做四项准备：分别创建 `PT-SFT-CUDA` 与 `PT-RL-CUDA`；从对应环境启动 JupyterLab；为每次真实运行设置唯一 `RUN_LABEL`；保持页面 5.1 的默认模型、样本和候选矩阵。模型缓存可以共享，Python 包环境不能混用。两个 Profile 均以 8 GiB 作为环境准入线；`PT-RL-CUDA` 的默认矩阵按 12 GB 级单卡准备。准入线不是实测峰值承诺，显存不足时记录失败，不同时缩小多个工作负载变量。

| 顺序 | Profile / 页面 | 本步验证什么 | 进入下一步的条件 |
|:---|:---|:---|:---|
| 0 | `PT-SFT-CUDA` 安装与预检 | PyTorch、CUDA、Transformers、PEFT、Datasets 与仓库运行时 | `pip check` 无冲突，Profile 返回 `status=ok` |
| 1 | Part 02 · 14 LoRA | 真实模块命中、冻结范围、低秩梯度与合并等价 | 三组目标模块均命中，基座冻结，合并误差过门槛 |
| 2 | Part 02 · 31 LoRA 变体 | 固定秩与固定参数预算下的 LoRA、rsLoRA、DoRA 路径 | 候选参数预算可比，执行路径与失败均已记录 |
| 3 | Part 02 · 15 端到端微调 | 数据提交版本、监督词元、训练/验证隔离与有限更新 | 固定切分可复现，更新前后损失和资源记录完整 |
| 4 | Part 02 · 30 长上下文微调 | 长度阶梯，以及固定样本的 padding / 边界隔离 Packing 对照 | 三档长度使用相同模型和测量方法；Packing 的监督 Token、标签边界、位置编号和跨样本 Attention 检查均通过 |
| 5 | `PT-RL-CUDA` 安装与预检 | 偏好与奖励实验所需的单卡隔离环境 | `pip check` 无冲突，Profile 返回 `status=ok` |
| 6 | Part 02 · PT-REWARD-SIGNAL | 策略偏移、终止奖励、KL 系数与回报折扣矩阵 | 行为策略偏移非零，终止奖励位置和候选矩阵完整 |
| 7 | Part 02 · 17 DPO | 固定参考策略、成对偏好边际和 LoRA 有限更新 | 三档 β 使用同一偏好对，偏好方向与梯度隔离成立 |
| 8 | Part 02 · 18 GRPO | 验证器奖励、组内优势、候选等权与策略分离 | 最小候选组与扩展候选组都形成正确更新方向；两档只比较候选覆盖，不用于选择组规模 |

## 机制页职责与学习级别

机制页不构成一条必须全部运行的线性路径。共同基础建立可复用语义，主路径形成最小闭环，分支页只在对应问题出现时进入。

| 页面 | 职责 | 学习级别 | GPU 状态 |
|:---|:---|:---|:---|
| [Part 02 · 33 微调准入](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb) | 根据失败证据选择 prompt、RAG、SFT 或偏好优化 | 共同入口 | 不需要 |
| [Part 02 · 32 SFT 数据准入](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb) | 检查样本身份、去重、去污染和长度分布 | SFT 主路径 | 不需要 |
| [Part 02 · 13 SFT 训练循环](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.ipynb) | 把 chat template、labels 与有效监督 Token 接入 loss | SFT 主路径 | 不需要；由 Part 02 · 15 承接真实训练 |
| [Part 02 · 14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.ipynb) | 建立目标模块、秩、缩放与适配器生命周期 | SFT 主路径 | `rtx5070ti_lora_targets_v3` 已完成；训练机制通过，最高峰值分配约 2.02 GiB；`attention_qv` 的 BF16 合并保持 `tune` |
| [Part 02 · 15 端到端微调](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.ipynb) | 验证训练、验证、检查点与交付闭环 | SFT 主路径 | `rtx5070ti_sft_v1` 已通过；峰值分配约 4.66 GiB，`accept_mechanism` |
| [Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) | 解释 NF4、冻结基座与低资源适配器更新 | 低资源分支 | 复用 `Q-QLORA` |
| [Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb) | 比较序列长度、padding、边界隔离 Packing 和显存预算 | 长上下文分支 | `rtx5070ti_length_packing_v4` 已通过；最高峰值分配约 2.90 GiB，`accept_mechanism` |
| [Part 02 · 31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb) | 比较 rsLoRA、DoRA、秩与固定参数预算 | 参数适配分支 | `rtx5070ti_lora_variants_v2` 已完成；最高峰值分配约 1.97 GiB；固定预算 `dora_r7` 保持 `tune` |
| [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) | 固定偏好对、数据隔离和独立评测协议 | 偏好共同基础 | 不需要 |
| [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb) | 解释参考策略、`beta` 与成对偏好更新 | 离线偏好分支 | `rtx5070ti_dpo_v1` 已通过；最高峰值分配约 3.19 GiB，`accept_mechanism` |
| [Part 02 · PT-REWARD-MODEL 奖励模型训练、校准与过度优化](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb) | 解释成对奖励目标、留出校准和代理奖励过度优化 | 奖励优化共同基础 | CPU 机制页；真实奖励模型训练留给后续固定数据与质量协议 |
| [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb) | 统一奖励、回报、价值与优势的关系 | 奖励优化共同基础 | `rtx5070ti_reward_signal_v3` 已通过；峰值分配约 2.01 GiB，`accept_mechanism` |
| [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb) | 解释策略、参考策略、奖励模型、价值模型与资源状态 | PPO 分支 | 不增加机制级单卡实验 |
| [Part 02 · 18 GRPO 与可验证奖励](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.ipynb) | 解释候选组、验证器奖励、组内基线与受约束更新 | GRPO / RLVR 分支 | v3 `default3` 已通过；最高峰值分配约 3.06 GiB，`accept_mechanism`；扩展候选组暴露较高裁剪率，不用于自动选择组规模 |
| [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb) | 管理轨迹身份、策略版本、新鲜度与准入 | 在线系统共同基础 | 不需要 |
| [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb) / [Part 02 · 51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.ipynb) | 分别承担多阶段系统编排与在线偏好反馈 | 两条系统分支 | 留给多工作进程或项目验证 |

## 机制与项目接口

| 机制链 | 机制页产出 | 项目页补充 |
|:---|:---|:---|
| SFT 数据 → 标签 → 损失 | 13、32 | 64 的数据准入与 62 的真实训练交付 |
| LoRA / QLoRA → 适配器 | 14、26、31 | 31 输出固定秩 / 固定预算机制证据；63/65 承接质量、资源与产物准入 |
| 偏好对 → DPO | 17、50 | 84 的训练—评测—决策记录 |
| 偏好对 / 验证结果 → 奖励 → 价值 / 优势 → PPO 或 GRPO | 奖励模型页与奖励信号页为共同基础，16 与 18 为并列算法分支 | 85 的组采样、奖励与更新闭环 |
| 策略新鲜度 → RL 系统或在线偏好 | 轨迹生命周期页为共同基础，RL 系统页与 51 为并列系统分支 | 85/86 的版本、编排、回滚与在线成本证据 |

## PT-RL-CUDA 机制工作负载

三项机制实验共享模型快照和单卡环境；17、18 共享 LoRA 有限更新路径，但各自保留候选变量，奖励信号页只比较固定轨迹上的信号构造。固定小样本用于隔离机制差异，不能替代独立偏好评测或在线系统验证。

| 页面 / 工作负载 | 固定条件 | 受控候选 | 结果路径 | 当前状态 |
|:---|:---|:---|:---|:---|
| Part 02 · 17 / `pt-dpo-real-model-update` | Qwen2.5-0.5B-Instruct、4 组偏好对、LoRA、4 次更新 | `beta = 0.05 / 0.10 / 0.30` | `benchmarks/results/pt_dpo/` | `rtx5070ti_dpo_v1` 已通过；三档偏好准确率均为 1.0 |
| Part 02 · 18 / `pt-grpo-real-model-update` | Qwen2.5-0.5B-Instruct、2 个固定提示、`integer-answer-v1` 验证器、LoRA、4 次更新 | `minimal_pair` / `expanded_group` 候选覆盖条件 | `benchmarks/results/pt_grpo/` | v3 `default3` 已通过；两档均形成正确策略分离，扩展组用于暴露格式、不可判定与裁剪饱和风险 |
| Part 02 · PT-REWARD-SIGNAL / `pt-reward-signal-matrix` | Qwen2.5-0.5B-Instruct、固定提示与回复、固定奖励 | KL 系数、终止奖励放置与优势构造 | `benchmarks/results/pt_reward_signal/` | `rtx5070ti_reward_signal_v3` 已通过；6 组候选完整，终局奖励位置检查通过 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、baseline/candidate |
| 模型与数据 | model/tokenizer revision、chat template、dataset revision、split 与样本 hash |
| 训练 | dtype、有效 batch、学习率、更新次数、随机种子与策略版本 |
| 质量与安全 | loss、任务指标、win-rate/judge 一致性、安全与回滚门槛 |
| 资源与产物 | 峰值显存、训练步 / 轨迹采样时间、产物 / 适配器路径与哈希 |
| 结论 | `failure`、`evidence_level`、decision 与下一步 |
