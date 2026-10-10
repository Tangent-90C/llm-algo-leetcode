# 推理优化实验 Quickstart

本页只说明环境创建和进入方式。页面、工作负载、结果路径和证据状态见[实验资产表](./experiment_assets.md)；跨专题隔离规则和硬件快照见[GPU 环境资产表](../../gpu_environment_assets.md)。以下命令均在仓库根目录执行。

<a id="inf-cuda-mechanism-quickstart"></a>

## INF-CUDA-MECHANISM：单卡推理机制环境

这一环境用于真实模型的迭代成本、执行分派、显存和局部延迟对照，不安装 vLLM 或 SGLang，也不形成服务吞吐结论。

```bash
python tools/install_environment_profile.py \
  --profile INF-CUDA-MECHANISM \
  --env-name llm_inference_mechanism_cu128

conda activate llm_inference_mechanism_cu128
python -m jupyter lab
```

只检查安装计划时添加 `--dry-run`。进入 Notebook 后先确认 Kernel 为该环境，再运行 Profile 预检；只有预检通过后才把 `RUN_MODE` 从 `dry_run` 改为 `real_gpu`。

```bash
conda activate llm_inference_mechanism_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile INF-CUDA-MECHANISM
python -m jupyter lab
```

<a id="inf-vllm-quickstart"></a>

## INF-VLLM-011-CU128：vLLM 服务环境

66–72 的服务项目使用独立 vLLM Profile。不要先安装 `torch-cu128.txt`，也不要向机制环境追加 vLLM。

```bash
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_vllm_cu128

conda activate llm_vllm_cu128
python -m jupyter lab
```

SGLang 对照使用独立环境，不能与 vLLM 共享同一个 Python 环境。尚未登记稳定 Profile 的后端只作为可选候选，不进入默认复测链。

## 常见异常

| 现象 | 处理方式 |
|:---|:---|
| `pip check` 报依赖冲突 | 停止执行，按目标 Profile 新建环境，不在旧环境逐包覆盖 |
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| CUDA 或显存预检失败 | 保持 `RUN_MODE=dry_run`，记录失败，不绕过硬件门槛 |
| 模型或数据下载失败 | 检查代理与缓存；不要把代理地址、令牌写入 Notebook |
| 服务能够启动但没有稳定结果 | 只记录环境/能力预检，不升级为服务性能证据 |
