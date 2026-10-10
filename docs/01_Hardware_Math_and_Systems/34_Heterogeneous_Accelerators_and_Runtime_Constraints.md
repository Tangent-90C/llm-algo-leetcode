# 34. Heterogeneous Accelerators and Runtime Constraints | 异构加速器与运行时约束

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


同一模型候选在 CPU、GPU、NPU 或 TPU 上不一定具有相同执行路径。本节依次检查设备能力、artifact—runtime—operator 契约、阶段放置和证据等级，把峰值规格转成可复查的异构可行性判断。

**环境：** CPU-first；真实设备验证进入部署项目。
## 前置阅读

- [10 算力现状与替代方案](./10_Domestic_AI_Chips_Overview.md)
- [09 AI 编译器与图优化](./09_AI_Compilers_and_Graph_Optimization.md)
- [07 CPU 与 GPU 异构调度](./07_CPU_GPU_Heterogeneous_Scheduling.md)
## Q1：为什么峰值算力不能代表设备可用性？

设备是否适合目标 workload，取决于容量、精度、算子、runtime、通信和隔离条件。任何一项硬约束失败，都可能让峰值规格失去意义。

| 判断维度 | 需要确认的条件 | 常见误判 |
|:---|:---|:---|
| 内存与带宽 | 权重、状态和 workspace 能否驻留并被访问 | 内存更大就一定更快 |
| 精度与格式 | dtype 和 artifact metadata 是否被识别 | 文件可见就代表可加载 |
| 算子与编译 | 关键算子是否有目标实现 | 框架可导入就代表可执行 |
| runtime 与通信 | 驱动、backend、collective 是否闭环 | 多设备天然获得线性扩展 |

![异构设备的能力、runtime 与可行性 gate](../public/01_Hardware_Math_and_Systems/1f_hardware_adaptation_system_map.svg)

```python
DEVICE_PROFILES = {
    "cpu": {"memory_gb": 128, "precisions": {"fp32", "bf16"}, "ops": {"tokenize", "route", "postprocess"}, "runtimes": {"python"}, "collective": False},
    "gpu": {"memory_gb": 48, "precisions": {"fp16", "bf16", "int8"}, "ops": {"prefill", "decode", "rope", "attention"}, "runtimes": {"torch", "vllm"}, "collective": True},
    "npu": {"memory_gb": 64, "precisions": {"bf16", "int8"}, "ops": {"prefill", "decode", "attention"}, "runtimes": {"vendor-runtime"}, "collective": True},
}


def summarize_device_profile(profile):
    """把设备 profile 转成便于比较的基础账本。"""
    return {
        "memory_gb": profile["memory_gb"],
        "precision_count": len(profile["precisions"]),
        "operator_count": len(profile["ops"]),
        "runtime_count": len(profile["runtimes"]),
        "collective": profile["collective"],
    }


assert summarize_device_profile(DEVICE_PROFILES["gpu"])["operator_count"] == 4
```

## Q2：artifact、runtime 与算子怎样组成可执行契约？

加载兼容、请求兼容和执行兼容是三类不同检查。artifact 提供候选身份，runtime 决定环境能否启动，operator gate 确认关键路径存在，质量 gate 再判断输出能否进入性能比较。

| Gate | 最少检查项 | 通过后的含义 |
|:---|:---|:---|
| artifact | revision、format、precision、tokenizer | 候选身份完整 |
| runtime | driver、compiler、backend、依赖 | 环境能够启动 |
| operator | required ops、dtype、collective | 关键路径可执行 |
| quality | 固定样本、阈值、失败样例 | 可以进入 benchmark |

```python
REQUEST_PLAN = {
    "weight_and_workspace_gb": 40,
    "precision": "bf16",
    "required_ops": {"prefill", "decode", "attention"},
    "runtime": "vllm",
    "requires_collective": False,
    "stages": {"prepare": {"tokenize"}, "generate": {"prefill", "decode", "attention"}},
}


def check_device_feasibility(profile, plan):
    """逐项保存硬约束检查，避免总分掩盖失败原因。"""
    memory_ok = profile["memory_gb"] >= plan["weight_and_workspace_gb"]
    precision_ok = plan["precision"] in profile["precisions"]
    missing_ops = sorted(plan["required_ops"] - profile["ops"])
    runtime_ok = plan["runtime"] in profile["runtimes"]
    collective_ok = (not plan["requires_collective"]) or profile["collective"]
    feasible = memory_ok and precision_ok and not missing_ops and runtime_ok and collective_ok
    return {
        "feasible": feasible,
        "memory_ok": memory_ok,
        "precision_ok": precision_ok,
        "missing_ops": missing_ops,
        "runtime_ok": runtime_ok,
        "collective_ok": collective_ok,
    }


assert check_device_feasibility(DEVICE_PROFILES["gpu"], REQUEST_PLAN)["feasible"]
assert not check_device_feasibility(DEVICE_PROFILES["npu"], REQUEST_PLAN)["runtime_ok"]
```

## Q3：为什么要按阶段而不是按整个模型选择设备？

解析、预处理、Prefill、Decode、状态保存和输出具有不同的控制流、算子与搬运条件。按阶段检查可以让 CPU 承担控制密集任务，让 accelerator 承担张量计算，同时显式暴露阶段交接。

| 阶段 | 常见候选 | 首要约束 |
|:---|:---|:---|
| 解析、tokenization、路由 | CPU / host | 控制流与队列等待 |
| 预处理或编码 | CPU / accelerator | 算子覆盖与输入搬运 |
| Prefill / Decode | accelerator | 权重、状态、kernel 与 runtime |
| 缓存与输出 | device / host / network | 容量、迁移、回压与回退 |

```python
def assign_stages(profiles, plan):
    """按阶段需求重算设备候选，并保留每个设备的失败原因。"""
    stage_trace = {}
    for stage, stage_ops in plan["stages"].items():
        stage_runtime = "python" if stage == "prepare" else plan["runtime"]
        stage_plan = dict(plan, required_ops=set(stage_ops), runtime=stage_runtime)
        checks = {name: check_device_feasibility(profile, stage_plan) for name, profile in profiles.items()}
        stage_trace[stage] = {
            "feasible_devices": [name for name, check in checks.items() if check["feasible"]],
            "checks": checks,
        }
    return stage_trace


STAGE_TRACE = assign_stages(DEVICE_PROFILES, REQUEST_PLAN)
assert "cpu" in STAGE_TRACE["prepare"]["feasible_devices"]
assert STAGE_TRACE["generate"]["feasible_devices"] == ["gpu"]
```

## Q4：可执行候选何时进入真实 benchmark？

硬约束失败直接拒绝；质量失败也不能用速度掩盖。只有阶段可执行且质量通过后，候选才进入目标 backend benchmark。CPU 账本或模拟结果能够筛选候选，但不能证明真实性能。

| 状态 | 决策 | 下一步 |
|:---|:---|:---|
| 某阶段无可执行设备 | `reject` | 修正 artifact、runtime 或算子约束 |
| 质量 gate 失败 | `reject` | 检查模型、模板或精度候选 |
| 只有账本或模拟证据 | `tune` | 在目标设备与 workload 上复测 |
| 质量与真实 backend 证据齐全 | `accept` | 进入服务拓扑和成本比较 |

```python
def deployment_recommendation(stage_trace, quality_gate_passed, evidence_level):
    """根据硬约束、质量与证据等级形成部署前建议。"""
    has_hard_failure = any(not detail["feasible_devices"] for detail in stage_trace.values())
    if has_hard_failure:
        return {"status": "reject", "reason": "至少一个阶段没有可执行设备"}
    if not quality_gate_passed:
        return {"status": "reject", "reason": "质量 gate 未通过"}
    if evidence_level != "real_backend_benchmark":
        return {"status": "tune", "reason": "需要目标 backend 的真实证据"}
    return {"status": "accept", "reason": "执行、质量与 backend 证据齐全"}


assert deployment_recommendation(STAGE_TRACE, True, "cpu_simulation")["status"] == "tune"
assert deployment_recommendation(STAGE_TRACE, True, "real_backend_benchmark")["status"] == "accept"
```

## 相关阅读

- [部署与异构系统专题](../topic_discussion/deployment_heterogeneous/intro.md)
- [通信与并行专题](../topic_discussion/communication_parallelism/intro.md)
- [Part 02 · 72 多模型部署比较](../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.md)