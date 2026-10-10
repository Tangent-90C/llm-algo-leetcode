# PERF-TRACE. Trace Dependencies and Critical Path | Trace 依赖与关键路径

**难度：** Medium　| 　**环境：** CPU 可完成机制练习；单卡 GPU 用于生成真实 Trace　| 　**标签：** `性能分析` `Trace` `关键路径`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

Trace 记录事件在时间线上的位置，但“耗时最长”不等于“决定总耗时”。本节从事件区间、资源串行关系和显式依赖出发，计算关键路径和暴露等待，把 CPU / GPU 时间线转成可验证的瓶颈候选。
## 前置阅读

- [Part 00 · 17 PyTorch Profiling 基础](../00_Prerequisites/17_PyTorch_Profiling_Basics.md)
- [Part 01 · 07 CPU/GPU 异构调度](../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.md)
### Step 1：从指标进入事件时间线

指标只说明快慢和资源量，Trace 还需要记录事件的开始、结束、资源和依赖。必须先统一时钟与单位，再讨论重叠、空洞和同步。

| 证据 | 核心字段 | 能回答 | 不能单独证明 |
| --- | --- | --- | --- |
| 指标 | 延迟、吞吐、显存 | 变化幅度 | 事件依赖 |
| Trace 事件 | `start`、`end`、`resource` | 事件何时发生 | 热点是否位于关键路径 |
| 依赖边 | `predecessors` | 谁在等待谁 | 候选改动后的真实收益 |

![Trace 证据层次](../public/02_PyTorch_Algorithms/perf_trace_evidence_layers.svg)
### Step 2：依赖图决定哪些时间可以重叠

同一资源上的事件通常串行；不同资源上的事件只有在没有数据依赖时才能重叠。因此关键路径是有向无环图上的最长依赖路径，不是所有事件耗时的简单求和。

| 边类型 | 来源 | 对重叠的约束 | Trace 现象 |
| --- | --- | --- | --- |
| 数据依赖 | 输出被后续事件消费 | 后继不得早于前驱完成 | 等待 event / fence |
| 资源串行 | 同一 CPU 线程或 CUDA stream | 同资源按时间顺序执行 | 连续的同色区间 |
| 全局同步 | 显式 synchronize 或隐式阻塞 | 所有相关工作在边界前收敛 | CPU 空洞或 GPU 断层 |
### Step 3：关键路径与暴露等待

事件即使耗时很长，如果完全被其他工作隐藏，就不直接延长本次关键路径。暴露等待是事件完成时间超出前驱就绪时间的部分，它才是后续优化应优先解释的空洞。

| 对象 | 计算口径 | 优先检查 | 结论边界 |
| --- | --- | --- | --- |
| 事件耗时 | `end - start` | 单事件成本 | 不代表关键性 |
| 关键路径 | 最长依赖路径 | 决定完成时间的事件链 | 受 Trace 完整性约束 |
| 暴露等待 | 依赖就绪后仍未执行的间隔 | 排队、同步、I/O 与通信 | 需要判别实验验证归因 |

![关键路径与暴露等待](../public/02_PyTorch_Algorithms/perf_trace_critical_path.svg)
### Step 4：实现 Trace 规范化、依赖图与关键路径

题目给定事件标识、时间区间、资源和显式前驱。你需要先拒绝无效事件，再补齐同资源串行边，最后在无环依赖图上计算最长路径和暴露等待。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1 · `normalize_trace_events` | 统一事件时间和持续时间 | 标识唯一；结束不早于开始；前驱存在 | 重复标识、负持续时间和未知前驱 |
| TODO 2 · `build_trace_dependencies` | 合并显式依赖与同资源串行边 | 边从前驱指向后继；同资源按时间顺序串行 | 跨资源依赖、同资源顺序与无重复边 |
| TODO 3 · `critical_path_report` | 计算关键路径和暴露等待 | 依赖图必须无环；返回路径、总时间和逐事件等待 | 已知 DAG、并行分支与环检测 |

```python
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple
```


```python
def normalize_trace_events(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Validate trace events and add a duration field in milliseconds."""
    # TODO 1（事件契约）：校验标识、时间区间和显式前驱。
    # normalized = ???  # 按 `start_ms` 和 `event_id` 排序的事件列表。
    raise NotImplementedError("TODO 1：请规范化 Trace 事件")


def build_trace_dependencies(events: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    """Combine explicit predecessors with same-resource serialization edges."""
    # TODO 2（依赖边）：合并显式边与同资源相邻事件边。
    # edges = ???  # `(predecessor, successor)` 去重后的边列表。
    raise NotImplementedError("TODO 2：请建立 Trace 依赖")


def critical_path_report(
    events: Sequence[Dict[str, object]], edges: Sequence[Tuple[str, str]]
) -> Dict[str, object]:
    """Return the longest dependency path and exposed waits for a trace DAG."""
    # TODO 3（关键路径）：拓扑遍历 DAG，累计最长完成时间。
    # critical_path = ???  # 决定最终完成时间的事件 ID 链。
    # exposed_wait_ms = ???  # 每个事件相对前驱就绪时间的暴露等待。
    raise NotImplementedError("TODO 3：请计算关键路径")
```

### 测试

```python
def _trace_fixture():
    """Return a trace with one CPU branch and one dependent GPU chain."""
    return [
        {"event_id": "load", "resource": "cpu", "start_ms": 0.0, "end_ms": 2.0, "predecessors": []},
        {"event_id": "prepare", "resource": "cpu", "start_ms": 2.5, "end_ms": 4.0, "predecessors": ["load"]},
        {"event_id": "h2d", "resource": "copy", "start_ms": 4.0, "end_ms": 6.0, "predecessors": ["prepare"]},
        {"event_id": "compute", "resource": "gpu", "start_ms": 7.0, "end_ms": 12.0, "predecessors": ["h2d"]},
        {"event_id": "log", "resource": "cpu", "start_ms": 4.2, "end_ms": 5.0, "predecessors": ["prepare"]},
    ]


def test_normalize_trace_events():
    """Validate ordering, duration calculation, and malformed-event rejection."""
    events = normalize_trace_events(list(reversed(_trace_fixture())))
    assert [event["event_id"] for event in events] == ["load", "prepare", "h2d", "log", "compute"]
    assert events[0]["duration_ms"] == 2.0
    bad = _trace_fixture() + [{"event_id": "bad", "resource": "cpu", "start_ms": 3.0, "end_ms": 2.0, "predecessors": []}]
    try:
        normalize_trace_events(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("负持续时间必须拒绝")


def test_trace_dependencies():
    """Validate explicit edges and same-resource serialization."""
    events = normalize_trace_events(_trace_fixture())
    edges = set(build_trace_dependencies(events))
    assert ("load", "prepare") in edges
    assert ("prepare", "h2d") in edges
    assert ("prepare", "log") in edges


def test_critical_path_report():
    """Validate critical path, total time, and exposed waits."""
    events = normalize_trace_events(_trace_fixture())
    report = critical_path_report(events, build_trace_dependencies(events))
    assert report["critical_path"] == ["load", "prepare", "h2d", "compute"]
    assert report["critical_path_ms"] == 12.0
    assert report["exposed_wait_ms"]["prepare"] == 0.5
    assert report["exposed_wait_ms"]["compute"] == 1.0


def run_perf_trace_tests():
    """Run all trace dependency and critical-path tests."""
    test_normalize_trace_events()
    test_trace_dependencies()
    test_critical_path_report()
    print("✅ Trace 规范化、依赖图与关键路径测试通过")


try:
    run_perf_trace_tests()
except NotImplementedError as error:
    print(f"⏳ {error}")
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难，再向下查看参考代码与解析。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple


# TODO 1：统一事件时间和持续时间。
def normalize_trace_events(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Validate trace events and add a duration field in milliseconds."""
    raw_ids = [str(event.get("event_id", "")) for event in events]
    if any(not event_id for event_id in raw_ids) or len(raw_ids) != len(set(raw_ids)):
        raise ValueError("event_id must be non-empty and unique")
    known_ids = set(raw_ids)
    normalized = []
    for event in events:
        start = float(event["start_ms"])
        end = float(event["end_ms"])
        if end < start:
            raise ValueError("event end must not precede start")
        predecessors = [str(item) for item in event.get("predecessors", [])]
        if any(item not in known_ids for item in predecessors):
            raise ValueError("unknown predecessor")
        normalized.append({
            **event,
            "event_id": str(event["event_id"]),
            "resource": str(event["resource"]),
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start,
            "predecessors": predecessors,
        })
    return sorted(normalized, key=lambda event: (event["start_ms"], event["event_id"]))


# TODO 2：合并显式依赖与同资源串行边。
def build_trace_dependencies(events: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    """Combine explicit predecessors with same-resource serialization edges."""
    edges = {
        (str(predecessor), str(event["event_id"]))
        for event in events
        for predecessor in event.get("predecessors", [])
    }
    by_resource = defaultdict(list)
    for event in events:
        by_resource[str(event["resource"])].append(event)
    for resource_events in by_resource.values():
        ordered = sorted(resource_events, key=lambda event: (float(event["start_ms"]), str(event["event_id"])))
        edges.update((str(left["event_id"]), str(right["event_id"])) for left, right in zip(ordered, ordered[1:]))
    return sorted(edges)


# TODO 3：计算最长依赖路径和暴露等待。
def critical_path_report(
    events: Sequence[Dict[str, object]], edges: Sequence[Tuple[str, str]]
) -> Dict[str, object]:
    """Return the longest dependency path and exposed waits for a trace DAG."""
    event_map = {str(event["event_id"]): event for event in events}
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    indegree = {event_id: 0 for event_id in event_map}
    for left, right in set(edges):
        if left not in event_map or right not in event_map:
            raise ValueError("edge references unknown event")
        successors[left].append(right)
        predecessors[right].append(left)
        indegree[right] += 1

    queue = deque(sorted(event_id for event_id, degree in indegree.items() if degree == 0))
    order = []
    while queue:
        event_id = queue.popleft()
        order.append(event_id)
        for successor in sorted(successors[event_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(order) != len(event_map):
        raise ValueError("trace dependency graph must be acyclic")

    completion = {}
    parent = {}
    exposed_wait = {}
    for event_id in order:
        ready_at = max((completion[item] for item in predecessors[event_id]), default=0.0)
        event = event_map[event_id]
        exposed_wait[event_id] = max(float(event["start_ms"]) - ready_at, 0.0)
        completion[event_id] = ready_at + exposed_wait[event_id] + float(event["duration_ms"])
        if predecessors[event_id]:
            parent[event_id] = max(predecessors[event_id], key=lambda item: completion[item])

    tail = max(completion, key=completion.get)
    path = []
    while True:
        path.append(tail)
        if tail not in parent:
            break
        tail = parent[tail]
    path.reverse()
    return {
        "critical_path": path,
        "critical_path_ms": completion[path[-1]],
        "exposed_wait_ms": exposed_wait,
    }
```

### 解析

**TODO 1：事件契约**

- 先检查事件标识和时间区间，再建立依赖，避免把脏 Trace 误判为性能现象。

**TODO 2：依赖边**

- 显式前驱表达数据依赖，同资源相邻边表达执行资源的串行约束。

**TODO 3：关键路径**

- 拓扑遍历同时完成环检测和最长路径动态规划；暴露等待保留了依赖就绪后仍未执行的时间。
### Step 5：GPU 机制实验——生成并读取真实 Trace

CPU 练习验证依赖图和关键路径算法；GPU 实验用 `torch.profiler` 记录 CPU 准备、H2D 和 CUDA 计算，确认实际同步与空洞。

| 对比项 | CPU 机制练习 | GPU 实验 |
| --- | --- | --- |
| 输入 | 手工事件与依赖 | Profiler 原始事件 |
| 核心证据 | 关键路径、暴露等待 | CPU / CUDA 时间线、同步、Trace artifact |
| 结论边界 | 算法与字段正确 | 本机工作负载的真实执行路径 |

#### 5.1 配置环境与固定工作负载

使用 `PERF-CUDA-MECHANISM` Profile；安装与预检命令见 [GPU 环境资产表](../gpu_environment_assets.md#perf-cuda-mechanism)。

```python
RUN_MODE = "dry_run"  # 默认值 dry_run；改为 real_gpu 后生成真实 Trace。
RUN_LABEL = "default"  # 默认值 default；正式复测使用唯一标签。
ENVIRONMENT_PROFILE = "PERF-CUDA-MECHANISM"  # 预检必须通过的环境标识。
WORKLOAD_ID = "perf-trace-critical-path"  # 稳定工作负载标识。
WORKLOAD_VERSION = "v1"  # Trace 字段、形状或测量协议变化时递增。
MATRIX_SIZE = 2048  # 默认矩阵边长；显存不足时下调。
WARMUP_STEPS = 5  # 不记入结果的预热次数。
PROFILE_STEPS = 10  # 进入 Trace artifact 的正式迭代数。
```

#### 5.2 执行并保存 Trace 与摘要 JSON

```python
from pathlib import Path
import json
import sys
import torch

if RUN_MODE == "real_gpu":
    if not torch.cuda.is_available():
        raise RuntimeError("PERF-TRACE 真实实验需要 CUDA GPU")
    from torch.profiler import ProfilerActivity, profile, record_function
    from tools.project_runtime import resolve_project_root

    device = torch.device("cuda")
    left = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    right = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    for _ in range(WARMUP_STEPS):
        _ = left @ right
    torch.cuda.synchronize()

    result_dir = resolve_project_root() / "benchmarks/results/perf_trace"
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_path = result_dir / f"{WORKLOAD_ID}_{RUN_LABEL}.trace.json"
    result_path = result_dir / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(PROFILE_STEPS):
            with record_function("prepare_cpu"):
                host_scale = torch.tensor(1.0)
            with record_function("cuda_matmul"):
                output = (left @ right) * host_scale.item()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    averages = prof.key_averages()
    record = {
        "schema_version": "performance-trace-mechanism/v1",
        "environment_profile": ENVIRONMENT_PROFILE,
        "workload": {
            "id": WORKLOAD_ID,
            "version": WORKLOAD_VERSION,
            "matrix_size": MATRIX_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "profile_steps": PROFILE_STEPS,
        },
        "framework": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
        "hardware": {"device": torch.cuda.get_device_name(0)},
        "metrics": {
            "cpu_time_total_us": sum(item.cpu_time_total for item in averages),
            "cuda_time_total_us": sum(getattr(item, "device_time_total", 0.0) for item in averages),
            "event_groups": len(averages),
        },
        "artifacts": {"trace_path": str(trace_path)},
        "failure": None,
        "evidence_level": "local_mechanism_trace",
        "decision": "ready_for_trace_reading",
    }
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    GPU_RESULT_PATH = result_path
else:
    print("⏭️ RUN_MODE=dry_run；未执行 GPU Trace。")
```

#### 5.3 读取结果

```python
if RUN_MODE == "real_gpu":
    trace_record = json.loads(Path(GPU_RESULT_PATH).read_text(encoding="utf-8"))
    print(json.dumps(trace_record, ensure_ascii=False, indent=2))
else:
    trace_record = {"status": "not_run", "decision": "enable_real_gpu_after_profile_preflight"}
    print(trace_record)
```

#### 5.4 解释 Trace 证据

| 观察 | 含义 | 下一步 |
| --- | --- | --- |
| CPU 空洞与同步边界对齐 | GPU 工作可能被主机同步暴露 | 检查 `.item()`、显式 synchronize 和输入依赖 |
| CUDA 事件密集但总时间仍长 | 需进一步用 Profile / Kernel 证据归因 | 进入性能分析 Task3–4 |
| Trace 与摘要 JSON 都完整 | 当前运行可用于关键路径阅读 | 保留 artifact 并登记环境与 workload |
## 相关阅读

- [性能分析 · Task2 时间线与关键路径](../topic_discussion/performance_analysis/intro.md)
- [Part 01 · 17 CUDA Stream 与异步](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)
- [PERF-IO · I/O 关键路径](./PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md)