# 后训练优化实验 Quickstart

本页只回答“如何创建或重新进入后训练 GPU 环境”。实验页面、模型、数据、结果路径和证据状态见[实验资产表](./experiment_assets.md)，精确 Profile 定义与跨专题隔离规则见[GPU 环境资产表](../../gpu_environment_assets.md)。以下命令均在仓库根目录执行。

后训练机制页使用两个专题 GPU Profile：`PT-SFT-CUDA` 负责 SFT 与 LoRA，`PT-RL-CUDA` 负责 DPO 与 GRPO。QLoRA 复用量化专题的 `Q-QLORA`；多 worker rollout 由项目单独登记，不属于本页的单卡环境。

<a id="pt-sft-cuda-quickstart"></a>

## PT-SFT-CUDA：SFT 与 LoRA 单卡环境

14、15、30、31 使用这一环境，分别验证 LoRA 挂载、真实 SFT 更新、长上下文资源曲线与 LoRA 变体预算对齐。它安装 Transformers、Datasets、Accelerate 与 PEFT，不与量化 Serving、vLLM 或分布式训练环境混装。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile PT-SFT-CUDA \
  --env-name llm_pt_sft_cu128
conda activate llm_pt_sft_cu128
python -m jupyter lab
```

安装器会创建隔离环境、安装 Profile 依赖、注册仓库公共运行时、执行 `pip check` 与 Profile 预检，并注册同名 Jupyter Kernel。已有环境默认不会被覆盖；只检查安装计划时添加 `--dry-run`，确认旧环境属于同一 Profile 后才能添加 `--reuse`。

### 后续复测

```bash
conda activate llm_pt_sft_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile PT-SFT-CUDA
python -m jupyter lab
```

<a id="pt-rl-cuda-quickstart"></a>

## PT-RL-CUDA：偏好优化与 RL 单卡机制环境

17、18 的真实偏好更新使用这一环境；奖励信号页完成后也复用同一 Profile。环境安装 Transformers、PEFT、Datasets 与 Accelerate，页面自行保留 DPO、GRPO 和奖励信号的机制计算，避免把框架默认行为当作机制答案。17 扫描 DPO 的 `beta` 与偏好边际，18 比较 GRPO 的组大小、组内优势和零方差保护。第 16 节的多角色 PPO 资源链不使用单卡有限更新代替完整系统证据。

TRL、verl、OpenRLHF 与独立采样后端属于项目级框架候选，不是该单卡机制 Profile 的必需依赖。当前约束文件是待验证组合；只有 17、18 在干净环境完成真实 GPU 复测后，才能登记为已验证快照。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile PT-RL-CUDA \
  --env-name llm_pt_rl_cu128
conda activate llm_pt_rl_cu128
python -m jupyter lab
```

### 后续复测

```bash
conda activate llm_pt_rl_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile PT-RL-CUDA
python -m jupyter lab
```

进入 Notebook 后，保持模型、固定样本和候选矩阵不变，只将 `RUN_MODE` 从默认的 `dry_run` 改为 `real_gpu`，并为 `RUN_LABEL` 填写唯一标签。17、18 分别把结果写入 `benchmarks/results/pt_dpo/` 与 `benchmarks/results/pt_grpo/`。

该 Profile 只形成单卡偏好目标、固定候选组、奖励信号和有限更新证据。真实候选采样吞吐、多工作进程轨迹采样、独立推理后端和分布式学习器使用项目指定环境。

## Q-QLORA：低比特训练交叉环境

26、65 使用量化专题维护的 `Q-QLORA`。创建、预检和复测命令见[量化与压缩实验 Quickstart](../quantization_compression/quickstart.md#q-qlora-quickstart)，不要向 `PT-SFT-CUDA` 追加 bitsandbytes。

## Notebook 运行前检查

进入 JupyterLab 后选择与环境同名的 Kernel，并运行 `sys.executable` 确认解释器位于对应 Conda 环境。只有仓库运行时检查通过、`pip check` 无冲突、Profile 预检返回 `status=ok`，且页面 5.1 已固定模型、数据与 workload，才能将 `RUN_MODE` 改为 `real_gpu`。

## 常见异常入口

| 现象 | 处理方式 |
|:---|:---|
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| 数据下载受代理影响 | 保留 `httpx[socks]`，检查代理环境变量，不在 Notebook 写入凭据 |
| 显存不足 | 先降低长度、微批次或候选数；不要改变多个变量后继续与旧结果比较 |
| loss 下降但评测未改善 | 检查 labels、有效 token、数据切分和独立评测，不直接采用候选 |
| rollout 版本过旧 | 记录 policy version 与 freshness failure，不静默混入当前更新 |
