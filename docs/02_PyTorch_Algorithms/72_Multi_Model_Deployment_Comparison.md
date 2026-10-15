# 72. Multi Model Deployment Comparison | Single-GPU Deployment and Backend Comparison

**难度：** Hard | **环境：** CPU 完成候选审计与决策；NVIDIA 单 GPU 完成可选 backend benchmark | **标签：** 推理优化、模型部署、vLLM、SGLang、benchmark | **目标人群：** 想把开源模型部署为可复测服务的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

部署一个模型不只是启动一条命令：模型权重、量化 artifact、GPU 显存、backend 和请求负载共同决定服务能否稳定运行。本项目以单张 NVIDIA GPU 为约束，先建立模型与 backend 的可比较条件，再用同一文本 workload 比较 vLLM 和 SGLang，最后把可运行性、延迟、吞吐、显存和质量记录为部署决策。

默认路线从小型 Qwen smoke 开始；有足够单卡显存且具备已验证 artifact 时，再进入 Qwen 27B 的 backend 对照。DeepSeek 的 TP / EP、多卡服务和 PD 分离属于 [81 分布式推理项目](./81_Distributed_Inference_Project.md)。

**关键词：** `single GPU`, `artifact`, `vLLM`, `SGLang`, `TTFT`, `TPOT`, `deployment benchmark`
## 前置阅读

**导语：** 先理解推理服务如何记录 TTFT、TPOT、吞吐与失败状态，再进入本节把这些指标应用到一个具体的单卡部署选择中。量化 artifact 会改变显存预算，backend 会改变服务执行路径，两者都需要在固定请求条件下比较。

- [Part 02 · 66 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [Part 02 · 67 量化推理与部署项目](./67_Quantized_Inference_and_Deployment.md)
- [Part 02 · 22 vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [量化与压缩专题](../topic_discussion/quantization_compression/intro.md)
### Step 1：建立单卡部署的候选层级

单卡项目先回答两个问题：服务链路能否跑通，以及目标模型是否能在预留 KV Cache 后稳定运行。小模型用于验证环境，不替代 27B 的服务证据；27B 的低比特 artifact 才是本节的核心部署候选。

| 层级 | 候选对象 | 推荐单卡条件 | 学习目标 |
|:---|:---|:---|:---|
| smoke | 当前已验证的小型 Qwen | 16–24 GB | 跑通下载、启动、请求、JSON 和失败记录 |
| 核心对照 | Qwen3.6-27B 的已验证低比特 artifact | 48 GB 起步 | 固定文本 workload，对照 vLLM / SGLang |
| 扩展候选 | Qwen3.8-27B artifact 或图像/OCR 请求 | 48–80 GB | 评估升级模型、量化和多模态输入的取舍 |

这三层共享同一份请求协议，但只在相邻候选之间改变一个主要变量：模型 artifact、backend 或输入模态。
### Step 2：理解 artifact、显存与 backend 的关系

模型 ID 只能说明架构来源；真正进入服务的是一个具体 revision、权重格式和量化 artifact。artifact 决定权重占用，剩余显存再由 KV Cache、运行时工作区和并发请求共享。vLLM 与 SGLang 都需要同时确认模型架构、tokenizer、量化格式与运行时版本能够匹配。

| 部署对象 | 需要确认的事实 | 失配时的结果 |
|:---|:---|:---|
| 模型与 revision | model ID、commit / revision、tokenizer 与聊天模板 | load failed 或输出格式异常 |
| artifact | 权重格式、量化方法、校验来源和本地路径 | unsupported 或数值/质量回归 |
| 单卡容量 | 权重、KV Cache 预留、最大上下文、并发和显存上限 | OOM、吞吐抖动或长尾上升 |
| backend | vLLM / SGLang 版本、模型支持、启动参数 | 无法启动、请求失败或指标不可比 |
| 请求协议 | prompt 集、生成长度、temperature、并发与 warmup | 同一模型也无法做有效横比 |
### Step 3：固定服务 workload 与证据口径

正式比较时，G0 只验证环境，G1 建立目标 artifact 的 backend 基线，G2 在完全相同的模型、artifact、精度和请求条件下只替换 backend。这样读到的差异才可以解释为服务栈差异。

| 记录块 | G0 smoke | G1 backend baseline | G2 backend candidate |
|:---|:---|:---|:---|
| 模型角色 | 小模型，验证链路 | 目标 27B artifact | 与 G1 完全相同 |
| 变化项 | 不形成性能结论 | 选择 vLLM 或 SGLang 之一 | 只替换另一个 backend |
| 固定条件 | 单卡、文本请求、JSON 路径 | revision、artifact、dtype、上下文、请求集、并发、warmup、repeats | 与 G1 相同 |
| 核心指标 | load / request success、failure | TTFT、TPOT、E2E、吞吐、P95、peak memory、质量 | 同一指标与 G1 对照 |
| 证据等级 | `single_gpu_smoke` | `real_backend_benchmark` | `real_backend_benchmark` |

质量检查使用固定短回答/格式任务；它不是完整能力评测，但能避免把无法遵守输出格式的 artifact 当作纯性能候选。
### Step 4：CPU 候选审计与部署决策

题目区把真实实验前必须完成的三项判断写成 CPU 可验证的逻辑：配置是否完整、G1/G2 是否共享同一 workload、以及结果是否满足质量与延迟门槛。骨架保留报告字段与派生统计，TODO 只要求填写会改变部署结论的机制判断。

| TODO | 函数 | 机制职责 | 测试验证 |
|:---|:---|:---|:---|
| 1 | `build_single_gpu_case` | 审计模型、artifact、backend、显存预算与 workload 是否构成可执行候选 | 缺少 artifact 或容量不足时不能标为 ready |
| 2 | `compare_backend_pair` | 判断 baseline / candidate 是否使用同一模型条件，并计算服务指标差值 | backend 不同、其他核心条件相同才可比较 |
| 3 | `decide_single_gpu_deployment` | 联合质量、TTFT、显存和失败状态输出 accept / tune / reject | OOM、质量不足和延迟收益不足走不同分支 |

```python
from typing import Any, Dict, List

```


```python
# 题目区只挖空三项部署机制：候选可执行性、同条件 backend 对照、部署决策。
# 模型字段、指标字段和报告结构由骨架提供，避免把格式填写误当作机制练习。

def build_single_gpu_case(
    model_id: str, artifact: str, backend: str, workload: Dict[str, int],
    gpu_memory_gb: float, required_memory_gb: float,
) -> Dict[str, Any]:
    """Audit whether one model artifact is runnable under a single-GPU budget.

    ``required_memory_gb`` is the planned weight + runtime + KV-cache budget,
    not a measured peak-memory value.
    """
    # TODO 1：单卡候选可执行性。
    # model_ready = ???  # model_id、artifact、backend 都是非空字符串。
    # workload_ready = ???  # prompt_tokens、max_tokens、concurrency 都为正数。
    # memory_ready = ???  # gpu_memory_gb >= required_memory_gb。
    # status = ???  # 三项均满足为 'ready'；否则为 'invalid_config' 或 'insufficient_memory'。
    raise NotImplementedError('请先完成 TODO 1！')


def compare_backend_pair(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two backend records only after their deployment contract matches."""
    # TODO 2：同条件 backend 对照。
    # comparable_fields = ('model_id', 'artifact', 'revision', 'dtype', 'workload_key')。
    # mismatches = ???  # 收集上述字段不一致的字段名。
    # ttft_gain_ms = ???  # baseline['ttft_ms'] - candidate['ttft_ms']。
    # throughput_gain = ???  # candidate['throughput_tok_s'] - baseline['throughput_tok_s']。
    raise NotImplementedError('请先完成 TODO 2！')


def decide_single_gpu_deployment(
    comparison: Dict[str, Any], candidate: Dict[str, Any],
    min_quality: float, max_ttft_ms: float,
) -> Dict[str, str]:
    """Turn compatibility, service metrics and quality into a deployment decision."""
    # TODO 3：部署决策机制。
    # has_failure = ???  # candidate 的 failure 非空，或 status 不是 'load_success'。
    # quality_ok = ???  # candidate['quality'] >= min_quality。
    # latency_ok = ???  # candidate['ttft_ms'] <= max_ttft_ms。
    # decision = ???  # failure -> reject；可比且质量/延迟达标 -> accept；其余 -> tune。
    raise NotImplementedError('请先完成 TODO 3！')

```

### 测试

测试分别覆盖：单卡预算不足、backend 对照条件不一致、OOM/质量/延迟三类部署决策。它们验证 CPU 审计逻辑，不替代 Step 5 的真实服务测量。

```python
def test_single_gpu_case_contract():
    ready = build_single_gpu_case(
        'Qwen/Qwen3.6-27B', 'validated-q4-artifact', 'vllm',
        {'prompt_tokens': 128, 'max_tokens': 64, 'concurrency': 1}, 48, 42,
    )
    assert ready['status'] == 'ready'
    limited = build_single_gpu_case(
        'Qwen/Qwen3.6-27B', 'validated-q4-artifact', 'vllm',
        {'prompt_tokens': 128, 'max_tokens': 64, 'concurrency': 1}, 24, 42,
    )
    assert limited['status'] == 'insufficient_memory'


def test_backend_pair_contract():
    baseline = {
        'model_id': 'Qwen/Qwen3.6-27B', 'artifact': 'validated-q4-artifact',
        'revision': 'r1', 'dtype': 'auto', 'workload_key': 'text-128-64-c1',
        'ttft_ms': 120.0, 'throughput_tok_s': 42.0,
    }
    candidate = {**baseline, 'backend': 'sglang', 'ttft_ms': 95.0, 'throughput_tok_s': 50.0}
    comparison = compare_backend_pair(baseline, candidate)
    assert comparison['comparable'] is True
    assert comparison['ttft_gain_ms'] == 25.0
    assert comparison['throughput_gain'] == 8.0
    assert compare_backend_pair(baseline, {**candidate, 'dtype': 'bf16'})['comparable'] is False
    return comparison, candidate


def test_single_gpu_decision():
    comparison, candidate = test_backend_pair_contract()
    accepted = decide_single_gpu_deployment(
        comparison, {**candidate, 'status': 'load_success', 'quality': 0.9, 'failure': None}, 0.8, 100.0
    )
    assert accepted['decision'] == 'accept'
    assert decide_single_gpu_deployment(
        comparison, {**candidate, 'status': 'load_success', 'quality': 0.6, 'failure': None}, 0.8, 100.0
    )['decision'] == 'tune'
    assert decide_single_gpu_deployment(
        comparison, {**candidate, 'status': 'oom', 'quality': 0.9, 'failure': 'CUDA out of memory'}, 0.8, 100.0
    )['decision'] == 'reject'


test_single_gpu_case_contract()
test_single_gpu_decision()
print('✅ 72 单卡部署候选、backend 对照与决策逻辑通过。')

```

---

## 参考代码与解析

### 代码

```python
def build_single_gpu_case(
    model_id: str, artifact: str, backend: str, workload: Dict[str, int],
    gpu_memory_gb: float, required_memory_gb: float,
) -> Dict[str, Any]:
    """Audit whether one model artifact is runnable under a single-GPU budget.

    ``required_memory_gb`` is the planned weight + runtime + KV-cache budget,
    not a measured peak-memory value.
    """
    # TODO 1：单卡候选可执行性。
    model_ready = all(isinstance(value, str) and value.strip() for value in (model_id, artifact, backend))
    workload_ready = isinstance(workload, dict) and all(workload.get(key, 0) > 0 for key in ('prompt_tokens', 'max_tokens', 'concurrency'))
    memory_ready = gpu_memory_gb >= required_memory_gb
    if not model_ready or not workload_ready:
        status = 'invalid_config'
    elif not memory_ready:
        status = 'insufficient_memory'
    else:
        status = 'ready'
    return {
        'status': status, 'model_id': model_id, 'artifact': artifact, 'backend': backend,
        'workload': dict(workload), 'gpu_memory_gb': gpu_memory_gb,
        'required_memory_gb': required_memory_gb,
    }


def compare_backend_pair(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two backend records only after their deployment contract matches."""
    # TODO 2：同条件 backend 对照。
    comparable_fields = ('model_id', 'artifact', 'revision', 'dtype', 'workload_key')
    mismatches = [field for field in comparable_fields if baseline.get(field) != candidate.get(field)]
    ttft_gain_ms = baseline.get('ttft_ms', 0.0) - candidate.get('ttft_ms', 0.0)
    throughput_gain = candidate.get('throughput_tok_s', 0.0) - baseline.get('throughput_tok_s', 0.0)
    return {
        'comparable': not mismatches, 'mismatches': mismatches,
        'ttft_gain_ms': round(ttft_gain_ms, 3), 'throughput_gain': round(throughput_gain, 3),
    }


def decide_single_gpu_deployment(
    comparison: Dict[str, Any], candidate: Dict[str, Any],
    min_quality: float, max_ttft_ms: float,
) -> Dict[str, str]:
    """Turn compatibility, service metrics and quality into a deployment decision."""
    # TODO 3：部署决策机制。
    has_failure = bool(candidate.get('failure')) or candidate.get('status') != 'load_success'
    quality_ok = candidate.get('quality', 0.0) >= min_quality
    latency_ok = candidate.get('ttft_ms', float('inf')) <= max_ttft_ms
    if has_failure:
        return {'decision': 'reject', 'reason': 'backend_or_memory_failure', 'next_action': 'inspect_artifact_and_runtime'}
    if comparison.get('comparable') and quality_ok and latency_ok:
        return {'decision': 'accept', 'reason': 'comparable_quality_and_latency_pass', 'next_action': 'repeat_with_production_concurrency'}
    return {'decision': 'tune', 'reason': 'quality_latency_or_contract_needs_review', 'next_action': 'adjust_artifact_or_service_config'}

```

### 解析

**TODO 1：单卡候选可执行性**

- 候选是否可运行由配置完整性、请求负载和计划显存预算共同决定。
- `required_memory_gb` 是部署前的预算字段，真实 peak memory 必须在 Step 5 中记录。

**TODO 2：同条件 backend 对照**

- 比较 backend 前，模型、artifact、revision、dtype 与 workload 必须一致。
- TTFT 差值和吞吐差值分别保留，避免只用一个指标替代交互体验和批量能力。

**TODO 3：部署决策**

- OOM、加载失败或请求失败直接进入 `reject`，因为没有可用服务。
- 可比条件下质量和 TTFT 都达标才 `accept`；其余情况进入 `tune`，再检查 artifact、上下文、并发或 backend 参数。
### Step 5（可选）：单卡 GPU / backend 部署实验

#### 5.1 环境、输入与固定条件

本实验先完成 G0 的小模型 smoke，再以一个已验证的 Qwen 27B 低比特 artifact 进行 G1/G2 backend 对照。默认不开启服务；填写 artifact、启动命令和 GPU 条件后才运行。

| 实验组 | 模型角色 | 主要变量 | 输出 JSON |
|:---|:---|:---|:---|
| G0 | 小型 Qwen smoke | 验证下载、启动与请求链路 | `g0_smoke.json` |
| G1 | Qwen 27B artifact baseline | 选择 vLLM 或 SGLang 之一 | `g1_baseline.json` |
| G2 | 与 G1 相同的 27B artifact | 只替换另一个 backend | `g2_backend_candidate.json` |

```python
# 5.1：默认关闭。单卡项目不配置 TP / EP / PD 分离。
RUN_GPU_EXPERIMENT = False
GPU_MIN_MEMORY_GB = 16
RESULT_DIR = 'benchmarks/results/72_single_gpu_deployment'
WORKLOAD_PATH = None  # 可改为固定 JSONL；None 时使用 INLINE_PROMPT。
INLINE_PROMPT = '用两句话解释 KV Cache 的作用。'
NUM_PROMPTS = 8
MAX_TOKENS = 64
CONCURRENCY = 1
WARMUP = 2
REPEATS = 3
TEMPERATURE = 0.0

# G1/G2 必须使用完全相同的 model_id、revision、artifact、dtype 与 workload。
# 先填入已在当前单卡上验证可加载的 27B 低比特 artifact，再开启相应组。
DEPLOYMENT_GROUPS = {
    'G0': {
        'enabled': False, 'role': 'single_gpu_smoke', 'backend': 'vllm',
        'model_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'revision': 'main',
        'artifact': 'original_safetensors', 'dtype': 'bfloat16',
        'base_url': 'http://127.0.0.1:8000', 'server_command': [],
        'result_path': f'{RESULT_DIR}/g0_smoke.json',
    },
    'G1': {
        'enabled': False, 'role': 'backend_baseline', 'backend': 'vllm',
        'model_id': 'Qwen/Qwen3.6-27B', 'revision': 'PINNED_REVISION',
        'artifact': 'SET_VALIDATED_LOW_BIT_ARTIFACT', 'dtype': 'auto',
        'base_url': 'http://127.0.0.1:8000', 'server_command': [],
        'result_path': f'{RESULT_DIR}/g1_baseline.json',
    },
    'G2': {
        'enabled': False, 'role': 'backend_candidate', 'backend': 'sglang',
        'model_id': 'Qwen/Qwen3.6-27B', 'revision': 'PINNED_REVISION',
        'artifact': 'SET_VALIDATED_LOW_BIT_ARTIFACT', 'dtype': 'auto',
        'base_url': 'http://127.0.0.1:8001', 'server_command': [],
        'result_path': f'{RESULT_DIR}/g2_backend_candidate.json',
    },
}
print({'run': RUN_GPU_EXPERIMENT, 'groups': {key: value['role'] for key, value in DEPLOYMENT_GROUPS.items()}})

```

#### 5.2 环境启动检查

检查当前 Notebook 是否识别 NVIDIA CUDA、可用显存与 vLLM/SGLang 命令。预检不会下载模型或启动服务。

```python
# 5.2：单卡预检。
import importlib.util
import shutil
import torch

GPU_READY = torch.cuda.is_available() and torch.cuda.device_count() == 1
report = {
    'cuda_available': torch.cuda.is_available(),
    'visible_gpu_count': torch.cuda.device_count(),
    'vllm_installed': importlib.util.find_spec('vllm') is not None,
    'sglang_installed': importlib.util.find_spec('sglang') is not None,
    'vllm_command': shutil.which('vllm'),
    'sglang_command': shutil.which('sglang'),
}
if torch.cuda.is_available():
    report['device'] = torch.cuda.get_device_name(0)
    report['memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
print(report)
if RUN_GPU_EXPERIMENT and not GPU_READY:
    raise RuntimeError('72 需要恰好一张可见 NVIDIA GPU；请先调整 Notebook runtime。')

```

#### 5.3 配置实验组与 JSON 路径

服务由学习者在独立终端或受管环境中启动。每组填写对应的启动命令后，本 Notebook 只作为统一请求 client 与结果记录器；G1/G2 的核心字段不一致时不允许进入 backend 对照。

```python
# 5.3：检查 G1/G2 的同条件对照契约。
contract_fields = ('model_id', 'revision', 'artifact', 'dtype')
g1, g2 = DEPLOYMENT_GROUPS['G1'], DEPLOYMENT_GROUPS['G2']
contract_mismatches = [field for field in contract_fields if g1[field] != g2[field]]
if contract_mismatches:
    raise ValueError(f'G1/G2 不可比较，字段不一致：{contract_mismatches}')
for group_name, group in DEPLOYMENT_GROUPS.items():
    print({
        'group': group_name, 'role': group['role'], 'backend': group['backend'],
        'result_path': group['result_path'], 'command_configured': bool(group['server_command']),
    })

```

#### 5.4 执行请求并保存 JSON

先在每组的 `server_command` 中填入已验证的 vLLM 或 SGLang 启动命令，再将该组 `enabled` 设为 `True`。执行单元会调用统一 benchmark client，将每组独立保存；未配置命令或服务不可达会保留 failure JSON。

```python
# 5.4：默认关闭；每次只开启一个组，避免单卡上并存两个大模型服务。
import json
import subprocess
import sys
from pathlib import Path

if RUN_GPU_EXPERIMENT:
    project_root = next((path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'tools').is_dir()), Path.cwd())
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    for group_name, group in DEPLOYMENT_GROUPS.items():
        if not group['enabled']:
            continue
        output = project_root / group['result_path']
        output.parent.mkdir(parents=True, exist_ok=True)
        if not group['server_command']:
            failure = {
                'workload': {'source': WORKLOAD_PATH or 'inline_prompt', 'requests': NUM_PROMPTS, 'max_tokens': MAX_TOKENS, 'concurrency': CONCURRENCY},
                'model': {'name': group['model_id'], 'revision': group['revision'], 'artifact': group['artifact']},
                'hardware': {'single_gpu_required': True},
                'backend': {'engine': group['backend'], 'precision': group['dtype']},
                'metrics': {}, 'failure': 'server_command_not_configured',
                'evidence_level': 'configuration_only',
                'decision': {'status': 'tune', 'reason': 'fill_server_command_before_benchmark', 'next_action': 'start_backend'},
            }
            output.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{group_name}: 已写入配置提示 {output}')
            continue
        subprocess.run(group['server_command'], check=True)
        command = [
            sys.executable, 'tools/benchmark_inference_backend.py', '--project', '72',
            '--base-url', group['base_url'], '--model', group['model_id'], '--label', group_name,
            '--backend', group['backend'], '--dtype', group['dtype'], '--batch', '1',
            '--prompt', INLINE_PROMPT, '--num-prompts', str(NUM_PROMPTS), '--max-tokens', str(MAX_TOKENS),
            '--temperature', str(TEMPERATURE), '--concurrency', str(CONCURRENCY),
            '--warmup', str(WARMUP), '--repeats', str(REPEATS), '--output', str(output),
        ]
        if WORKLOAD_PATH:
            command.extend(['--workload', WORKLOAD_PATH])
        subprocess.run(command, cwd=project_root, check=True)
else:
    print('GPU/backend 实验未启动：完成 5.1–5.3 配置后，将 RUN_GPU_EXPERIMENT 设为 True。')

```

#### 5.5 读取结果与记录证据

先确认每组是否真的完成服务测量，再核对 G1/G2 的模型、artifact、revision、dtype 与 workload。下表中的结果应来自每组 JSON，而不是手填的估计值。

| 读取项 | G0 | G1 | G2 |
|:---|:---|:---|:---|
| 角色 | smoke | backend baseline | backend candidate |
| 状态 | load / request success 或 failure | real benchmark 或 failure | real benchmark 或 failure |
| 必要指标 | 成功率与 failure | TTFT、TPOT、E2E、吞吐、P95、peak memory、质量 | 与 G1 同一组指标 |
| evidence level | `single_gpu_smoke` 或 `configuration_only` | `real_backend_benchmark` | `real_backend_benchmark` |

```python
# 5.5：读取每组 JSON；不启动服务。
from pathlib import Path
import json

loaded_results = {}
for group_name, group in DEPLOYMENT_GROUPS.items():
    result_path = Path(group['result_path'])
    if not result_path.exists():
        print(f'{group_name}: 尚无结果 {result_path}')
        continue
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if 'normalized_result' in result:
        normalized = result['normalized_result']
        loaded_results[group_name] = normalized
        print({
            'group': group_name, 'strategy': normalized.get('strategy'),
            'evidence_level': normalized.get('evidence_level'), 'failure': normalized.get('failure'),
            'decision': normalized.get('decision'),
        })
    else:
        loaded_results[group_name] = result
        print({
            'group': group_name, 'evidence_level': result.get('evidence_level'),
            'failure': result.get('failure'), 'decision': result.get('decision'),
        })

```

#### 5.6 解释结果与形成部署决策

G0 成功只说明单卡链路可用；G1/G2 才能说明同一 27B artifact 在两个 backend 下的服务取舍。先处理 failure，再检查质量门槛，最后按交互目标比较 TTFT/P95 与吞吐。

| 观察到的结果 | 决策 | 下一步 |
|:---|:---|:---|
| artifact 无法加载、OOM 或服务请求失败 | `reject` | 记录失败，调整 artifact、上下文或显存预算 |
| G1/G2 条件不一致，或质量/延迟未同时满足目标 | `tune` | 固定 contract，调整一个 artifact 或 backend 参数后复测 |
| G1/G2 可比，且质量、TTFT、P95、吞吐与显存满足服务目标 | `accept` | 在同一单卡上增加并发与重复次数，形成部署档案 |
## 相关阅读

- [Part 02 · 66 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [Part 02 · 67 量化推理与部署项目](./67_Quantized_Inference_and_Deployment.md)
- [Part 02 · 81 分布式推理项目](./81_Distributed_Inference_Project.md)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [SGLang 官方文档](https://docs.sglang.ai/)
- [Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)