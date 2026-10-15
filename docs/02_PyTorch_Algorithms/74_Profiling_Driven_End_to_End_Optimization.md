# 74. Profiling Driven End to End Optimization | profiling 驱动的候选准入与交接

**难度：** Hard | **环境：** CPU 可完成报告收口；GPU 用于真实 trace 与候选复测 | **标签：** `性能优化`, `性能剖析`, `候选准入` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

当一次优化改变了训练时间或资源占用时，本节带你判断变化来自哪里，并用证据和复测决定候选能否进入综合项目。你将从已有结果提出瓶颈假设，再用针对性的证据和复测检查这个假设，最后生成供 Part 05 使用的性能优化交接包。

**关键词：** `profiling`, `optimization`, `end-to-end`

---

## 前置阅读

**导语：** 先了解 profiling 的基本观察方法，再结合训练测量、显存预算和策略比较结果，进入本节的端到端证据分析。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)

---
### Step 1：接收已选候选并固定回归范围
本项目不重新搜索优化方向，而是接收已经通过候选实验和预算筛选的方案。基线、候选、原瓶颈假设和主要负载共同构成输入；回归验收同时覆盖目标收益、质量与资源护栏，并加入至少一个相邻负载检查结论是否只在单点成立。

| 输入资产 | 验收操作 | 证据与交付物 |
|:---|:---|:---|
| 已固定的基线与候选 | 核对模型、数据、后端和唯一变化 | 可配对的运行契约 |
| 原瓶颈假设与候选理由 | 在相同阶段重新采集指标和 trace | 假设被支持、被否定或证据不足 |
| 主要负载与相邻负载 | 重复运行并检查质量、性能和资源护栏 | 配对样本与回归清单 |
| 回滚条件 | 汇总硬失败、回归与新瓶颈 | `accept`、`tune` 或 `reject` 的验收记录 |

![74 profiling 端到端闭环](../public/02_PyTorch_Algorithms/74_end_to_end_profile_loop.svg)

### Step 2：复测原瓶颈并检查代价转移

候选如果减少了原阶段耗时，还要确认收益是否被搬运、同步、重算、缓存维护或其他阶段抵消。总指标回答端到端是否改善，阶段 trace 回答原瓶颈是否减轻以及新瓶颈出现在哪里；两类证据必须来自同一运行契约。

| 证据层 | 要比较的内容 | 失败信号 |
|:---|:---|:---|
| 端到端 | 延迟、吞吐、峰值容量和质量 | 目标指标无收益或护栏回归 |
| 原瓶颈阶段 | 阶段总时间、调用次数和关键活动 | 阶段占比未下降或实际路径未生效 |
| 代价阶段 | 重算、搬运、同步、缓存维护与通信 | 新增成本抵消原收益 |
| 相邻负载 | 不同批次、序列长度或请求并发 | 候选只在单一负载点有效 |


### Step 3：形成准入、调优或回退结论

验收先处理运行失败、身份错配和关键缺证，再检查质量与资源护栏，最后判断端到端收益和瓶颈变化。候选通过时只表示它可以进入 Part 05 组合复测，不代表已经完成系统级交付。

| 状态 | 触发条件 | 后续动作 |
|:---|:---|:---|
| `reject` | 运行失败、配对失效、质量或安全护栏失败 | 回退基线并修复候选 |
| `tune` | 关键证据缺失、收益不稳定或代价转移仍可修复 | 补测或调整候选后重新验收 |
| `accept` | 主要负载收益达标，相邻负载无硬回归，原瓶颈减轻且资源可接受 | 固化候选和回归阈值 |

![74 对照实验与决策流程](../public/02_PyTorch_Algorithms/74_experiment_decision_flow.svg)

### Step 4：实现回归验收与采用决策

计时、指标差值和理论 Roofline 作为给定工具。请完成 3 个项目级任务：核验基线与候选是否可以配对，汇总主要负载与相邻负载的回归证据，再结合原瓶颈变化和新代价决定采用、调优或回退。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：核验回归对 | 已选候选、固定基线和唯一变化项 | 身份错配、缺少主要负载或重复负载时停止验收 | 正常与错配输入均有断言 |
| TODO 2：构造验收证据 | 主要与相邻负载的端到端指标、阶段 trace 和质量护栏 | 输出收益、回归、原瓶颈变化与新瓶颈 | 单点收益和相邻负载回归可区分 |
| TODO 3：形成采用状态 | 验收证据与最低收益、最大回归阈值 | 区分 `accept`、`tune`、`reject` 并给出回滚动作 | 硬失败优先于性能收益 |


```python
import math
import time

```


```python
import time

def estimate_roofline(flops, memory_bytes, peak_flops, bandwidth):
    """用给定的理论参数估算 Roofline 上限；不读取 GPU 硬件计数器。"""
    values = {
        'flops': flops, 'memory_bytes': memory_bytes,
        'peak_flops': peak_flops, 'bandwidth': bandwidth,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError('flops/memory_bytes/peak_flops/bandwidth 必须为正数')
    arithmetic_intensity = flops / memory_bytes
    compute_roof = peak_flops
    bandwidth_roof = bandwidth * arithmetic_intensity
    attainable_roof = min(compute_roof, bandwidth_roof)
    bound = 'compute' if compute_roof <= bandwidth_roof else 'memory'
    return {
        'arithmetic_intensity': arithmetic_intensity,
        'compute_roof': compute_roof,
        'bandwidth_roof': bandwidth_roof,
        'attainable_roof': attainable_roof,
        'bound': bound,
        'evidence_level': 'theoretical_formula_only',
    }

def benchmark_fn(fn, warmup=3, iters=10):
    """测量 CPU 函数的平均耗时；GPU trace 在 Step 5 独立采集。"""
    if warmup < 0 or iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_time_ms = total / iters * 1000
    return avg_time_ms

def summarize_optimization_result(base_metrics, tuned_metrics):
    """统一计算 baseline - tuned 的时间、显存和吞吐差异。"""
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    memory_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['throughput'] - base_metrics['throughput']

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    """输出指标变化、瓶颈证据边界和下一步动作。"""
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| step time | {summary['step_time_delta_ms']} ms | {'改善' if summary['time_improved'] else '未改善'} |",
        f"| peak memory | {summary['peak_mem_delta_mb']} MB | {'改善' if summary['memory_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
    ]
    conclusion = f"瓶颈判断：{bottleneck}。下一步：{next_action}。"
    return "\n".join([header, sep] + rows + [conclusion])


def recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0):
    """使用显式阈值输出 accept / tune / reject，不把 CPU 结果写成 GPU 结论。"""
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    strong_throughput_gain = summary['throughput_delta'] >= min_throughput_delta
    positive_resource_gain = summary['memory_improved'] or summary['throughput_improved']
    if strong_time_gain and (strong_memory_gain or strong_throughput_gain):
        decision = 'accept'
        reason = '时间达到阈值，且显存或吞吐至少一项达到阈值，当前优化值得保留。'
    elif strong_time_gain and positive_resource_gain:
        decision = 'tune'
        reason = '时间改善成立，但资源或吞吐收益还不够稳，建议继续微调。'
    else:
        decision = 'reject'
        reason = '当前优化没有形成稳定的端到端收益，建议回退或重新定位瓶颈。'
    return {'decision': decision, 'reason': reason}

def validate_regression_contract(contract, workload_records):
    """确认已选候选、固定基线和回归负载可以配对。"""
    # ==========================================
    # TODO 1：核验回归对。
    # 检查 baseline_id、candidate_id、changed_variable、主要负载数量和 workload_id 唯一性。
    # issues = ???
    # 返回 is_valid、issues、primary_workload。
    # ==========================================
    raise NotImplementedError('请先完成 TODO 代码！')

def build_regression_evidence(workload_records, original_bottleneck):
    """汇总主要与相邻负载的收益、护栏和瓶颈变化。"""
    # ==========================================
    # TODO 2：构造验收证据。
    # 对每个负载计算时间、显存、吞吐和质量差异，并比较原瓶颈阶段在候选中的变化。
    # rows = ???
    # hard_failures = ???
    # 返回 primary、adjacent、original_bottleneck_reduced 和 candidate_bottleneck。
    # ==========================================
    raise NotImplementedError('请先完成 TODO 代码！')

def decide_regression_adoption(evidence, gates):
    """根据端到端收益、相邻负载和瓶颈转移输出采用状态。"""
    # ==========================================
    # TODO 3：形成采用状态。
    # 硬失败或质量越界直接 reject；主要收益不足、相邻负载回归或瓶颈未减轻时 tune；其余 accept。
    # 返回 decision、reason、rollback 和 next_action。
    # ==========================================
    raise NotImplementedError('请先完成 TODO 代码！')

```

### 测试


```python
def test_optimization_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        result = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert result >= 0.0, "平均耗时应该非负"
        roofline = estimate_roofline(flops=2_000.0, memory_bytes=1_000.0, peak_flops=10.0, bandwidth=1.0)
        assert roofline['arithmetic_intensity'] == 2.0
        assert roofline['bound'] == 'memory'
        assert roofline['evidence_level'] == 'theoretical_formula_only'
        for invalid in ({'warmup': -1, 'iters': 2}, {'warmup': 0, 'iters': 0}):
            try:
                benchmark_fn(fn, **invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('非法 warmup / iters 应明确拒绝')

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0, 'throughput': 80.0}
        tuned = {'step_time_ms': 96.0, 'peak_mem_mb': 7168.0, 'throughput': 100.0}
        summary = summarize_optimization_result(baseline, tuned)

        assert summary['step_time_delta_ms'] == 24.0
        assert summary['peak_mem_delta_mb'] == 1024.0
        assert summary['throughput_delta'] == 20.0
        assert summary['time_improved'] is True
        assert summary['memory_improved'] is True
        assert summary['throughput_improved'] is True

        report = format_optimization_report(summary, 'backward kernel 占比过高', '保留混合精度并继续检查 optimizer')
        assert '| 指标 | 变化 | 判断 |' in report
        assert 'backward kernel 占比过高' in report
        assert '保留混合精度并继续检查 optimizer' in report

        decision = recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert decision['decision'] == 'accept'

        memory_summary = {'step_time_delta_ms': 12.0, 'peak_mem_delta_mb': 768.0, 'throughput_delta': 0.0, 'time_improved': True, 'memory_improved': True, 'throughput_improved': False}
        memory_decision = recommend_optimization_decision(memory_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert memory_decision['decision'] == 'accept'

        mixed_summary = {'step_time_delta_ms': 12.0, 'peak_mem_delta_mb': 128.0, 'throughput_delta': 2.0, 'time_improved': True, 'memory_improved': True, 'throughput_improved': True}
        mixed_decision = recommend_optimization_decision(mixed_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert mixed_decision['decision'] == 'tune'

        weak_summary = {'step_time_delta_ms': -3.0, 'peak_mem_delta_mb': 64.0, 'throughput_delta': 1.0, 'time_improved': False, 'memory_improved': True, 'throughput_improved': True}
        weak_decision = recommend_optimization_decision(weak_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert weak_decision['decision'] == 'reject'

        contract = {'baseline_id': 'train-base-v1', 'candidate_id': 'checkpoint-v2', 'changed_variable': 'activation_strategy'}
        workloads = [
            {
                'workload_id': 'b2-s1024', 'role': 'primary', 'status': 'ok',
                'baseline': {'step_time_ms': 120.0, 'peak_mem_mb': 11000.0, 'throughput': 16.0, 'quality': 1.00},
                'candidate': {'step_time_ms': 105.0, 'peak_mem_mb': 8200.0, 'throughput': 15.0, 'quality': 1.01},
                'baseline_stages_ms': {'forward': 30.0, 'backward': 65.0, 'update': 25.0},
                'candidate_stages_ms': {'forward': 34.0, 'backward': 45.0, 'update': 26.0},
            },
            {
                'workload_id': 'b1-s512', 'role': 'adjacent', 'status': 'ok',
                'baseline': {'step_time_ms': 70.0, 'peak_mem_mb': 7000.0, 'throughput': 18.0, 'quality': 0.98},
                'candidate': {'step_time_ms': 68.0, 'peak_mem_mb': 5600.0, 'throughput': 17.8, 'quality': 0.99},
                'baseline_stages_ms': {'forward': 20.0, 'backward': 35.0, 'update': 15.0},
                'candidate_stages_ms': {'forward': 22.0, 'backward': 31.0, 'update': 15.0},
            },
        ]
        contract_check = validate_regression_contract(contract, workloads)
        assert contract_check['is_valid'] is True and contract_check['primary_workload'] == 'b2-s1024'
        duplicate = validate_regression_contract(contract, workloads + [dict(workloads[1])])
        assert duplicate['is_valid'] is False
        evidence = build_regression_evidence(workloads, 'backward')
        assert evidence['original_bottleneck_reduced'] is True
        gates = {'min_primary_time_gain_ms': 10.0, 'max_adjacent_slowdown_ms': 3.0, 'max_quality_delta': 0.02}
        adoption = decide_regression_adoption(evidence, gates)
        assert adoption['decision'] == 'accept' and adoption['rollback'] is False
        regressed = [dict(workloads[0]), {**workloads[1], 'candidate': {**workloads[1]['candidate'], 'step_time_ms': 76.0}}]
        assert decide_regression_adoption(build_regression_evidence(regressed, 'backward'), gates)['decision'] == 'tune'
        failed = [{**workloads[0], 'status': 'failed'}]
        assert decide_regression_adoption(build_regression_evidence(failed, 'backward'), gates)['decision'] == 'reject'

        print("✅ profiling 驱动的端到端优化项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_optimization_project_template()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码


```python
import time


def estimate_roofline(flops, memory_bytes, peak_flops, bandwidth):
    """Estimate a theoretical Roofline bound; this does not collect hardware counters."""
    values = {
        'flops': flops, 'memory_bytes': memory_bytes,
        'peak_flops': peak_flops, 'bandwidth': bandwidth,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError('flops/memory_bytes/peak_flops/bandwidth must be positive')
    arithmetic_intensity = flops / memory_bytes
    compute_roof = peak_flops
    bandwidth_roof = bandwidth * arithmetic_intensity
    attainable_roof = min(compute_roof, bandwidth_roof)
    bound = 'compute' if compute_roof <= bandwidth_roof else 'memory'
    return {
        'arithmetic_intensity': arithmetic_intensity,
        'compute_roof': compute_roof,
        'bandwidth_roof': bandwidth_roof,
        'attainable_roof': attainable_roof,
        'bound': bound,
        'evidence_level': 'theoretical_formula_only',
    }

def benchmark_fn(fn, warmup=3, iters=10):
    if warmup < 0 or iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_time_ms = total / iters * 1000
    return avg_time_ms


def summarize_optimization_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    memory_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['throughput'] - base_metrics['throughput']

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| step time | {summary['step_time_delta_ms']} ms | {'改善' if summary['time_improved'] else '未改善'} |",
        f"| peak memory | {summary['peak_mem_delta_mb']} MB | {'改善' if summary['memory_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
    ]
    conclusion = f"瓶颈判断：{bottleneck}。下一步：{next_action}。"
    return "\n".join([header, sep] + rows + [conclusion])


def recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0):
    # TODO 3 对应变量 strong_time_gain：对照 min_time_delta_ms。
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    strong_throughput_gain = summary['throughput_delta'] >= min_throughput_delta
    positive_memory_gain = summary['peak_mem_delta_mb'] > 0
    positive_throughput_gain = summary['throughput_delta'] > 0
    # TODO 3 对应变量 positive_resource_gain：显存或吞吐任一出现正向变化。
    positive_resource_gain = positive_memory_gain or positive_throughput_gain
    if strong_time_gain and (strong_memory_gain or strong_throughput_gain):
        decision = 'accept'
        reason = '时间达到阈值，且显存或吞吐至少一项达到阈值，当前优化值得保留。'
    elif strong_time_gain and positive_resource_gain:
        decision = 'tune'
        reason = '时间改善成立，但资源或吞吐收益还不够稳，建议继续微调。'
    else:
        decision = 'reject'
        reason = '当前优化没有形成稳定的端到端收益，建议回退或重新定位瓶颈。'
    return {'decision': decision, 'reason': reason}

def validate_regression_contract(contract, workload_records):
    issues = []
    for key in ('baseline_id', 'candidate_id', 'changed_variable'):
        if not isinstance(contract.get(key), str) or not contract.get(key):
            issues.append(f'{key}_missing')
    primary = [item for item in workload_records if item.get('role') == 'primary']
    if len(primary) != 1:
        issues.append('primary_workload_count_invalid')
    workload_ids = [item.get('workload_id') for item in workload_records]
    if len(workload_ids) != len(set(workload_ids)):
        issues.append('duplicate_workload_id')
    return {'is_valid': not issues, 'issues': issues, 'primary_workload': primary[0]['workload_id'] if len(primary) == 1 else None}

def build_regression_evidence(workload_records, original_bottleneck):
    rows = []
    hard_failures = []
    for record in workload_records:
        if record.get('status') != 'ok':
            hard_failures.append(record.get('workload_id'))
            continue
        baseline = record['baseline']
        candidate = record['candidate']
        candidate_stages = record['candidate_stages_ms']
        row = {
            'workload_id': record['workload_id'], 'role': record['role'],
            'time_gain_ms': baseline['step_time_ms'] - candidate['step_time_ms'],
            'memory_gain_mb': baseline['peak_mem_mb'] - candidate['peak_mem_mb'],
            'throughput_gain': candidate['throughput'] - baseline['throughput'],
            'quality_delta': candidate['quality'] - baseline['quality'],
            'original_bottleneck_delta_ms': record['baseline_stages_ms'][original_bottleneck] - candidate_stages[original_bottleneck],
            'candidate_bottleneck': max(candidate_stages, key=candidate_stages.get),
        }
        rows.append(row)
    primary = next((row for row in rows if row['role'] == 'primary'), None)
    adjacent = [row for row in rows if row['role'] == 'adjacent']
    return {
        'primary': primary, 'adjacent': adjacent, 'hard_failures': hard_failures,
        'original_bottleneck': original_bottleneck,
        'original_bottleneck_reduced': primary is not None and primary['original_bottleneck_delta_ms'] > 0,
        'candidate_bottleneck': primary['candidate_bottleneck'] if primary else None,
    }

def decide_regression_adoption(evidence, gates):
    if evidence.get('hard_failures') or evidence.get('primary') is None:
        return {'decision': 'reject', 'reason': 'hard_failure_or_primary_missing', 'rollback': True, 'next_action': 'restore_baseline'}
    rows = [evidence['primary']] + evidence.get('adjacent', [])
    if any(row['quality_delta'] > gates['max_quality_delta'] for row in rows):
        return {'decision': 'reject', 'reason': 'quality_guardrail_failed', 'rollback': True, 'next_action': 'restore_baseline'}
    primary_gain_ok = evidence['primary']['time_gain_ms'] >= gates['min_primary_time_gain_ms']
    adjacent_ok = all(row['time_gain_ms'] >= -gates['max_adjacent_slowdown_ms'] for row in evidence.get('adjacent', []))
    if not primary_gain_ok or not adjacent_ok or not evidence.get('original_bottleneck_reduced', False):
        return {'decision': 'tune', 'reason': 'benefit_or_bottleneck_evidence_incomplete', 'rollback': False, 'next_action': 'adjust_and_rerun_regression'}
    return {'decision': 'accept', 'reason': 'benefit_and_guardrails_passed', 'rollback': False, 'next_action': 'freeze_candidate_and_thresholds'}

```

### 解析

本节接收已经通过候选实验和预算筛选的方案，只负责回归验收。计时、差值计算和理论 Roofline 是给定工具；学习者负责回归配对、瓶颈转移和采用判断。

**1. 机制：核验回归对**

- 基线、候选和唯一变化项必须明确；主要负载只能有一个，负载身份不能重复。
- 不变量：身份错配的运行不能进入回归验收。
- 测试证据：测试覆盖合法回归集合与重复负载。

**2. 机制：构造多负载验收证据**

- 主要负载决定目标收益，相邻负载检查候选是否只在单点有效；阶段耗时同时验证原瓶颈是否减轻以及新瓶颈落在哪里。
- 不变量：端到端变快不能掩盖质量越界或相邻负载明显回归。
- 测试证据：测试验证反向阶段缩短，并构造相邻负载变慢的反例。

**3. 机制：形成采用、调优或回退状态**

- 运行失败或质量越界时回退；主要收益不足、相邻负载回归或原瓶颈未减轻时继续调优；全部通过才采用。
- 不变量：硬失败和质量护栏优先于任何性能收益。
- 测试证据：测试覆盖采用、调优和回退三种状态。

**项目化原则**

- **一次只改一个变量**：否则收益不可归因。
- **主要与相邻负载一起复测**：避免只保留单点最优结果。
- **保留回滚条件**：采用记录必须能指出何时恢复基线。

### Step 5（项目实验）：核对性能证据并生成交接结果

本实验核对两组结论：完整训练配置能否采用 checkpoint，以及 checkpoint 在激活显存占主导时是否有效。两组实验的优化器和序列长度不同，因此分别判断。

| 实验范围 | 固定配置 | 可输出的结论 |
| --- | --- | --- |
| 完整训练 | 0.5B、FP32、AdamW、`seq_len=768` | `accept / tune / reject` |
| 激活显存 | 0.5B、FP32、无动量 SGD、长度阶梯 | `accept_for_activation_memory_scope / tune` |

#### 5.1 本次运行配置

只需填写 `RUN_LABEL` 和 `RUN_STAGE`。模型、工作负载、门槛、上游路径及证据哈希从 5.2 的固定清单读取。

| `RUN_STAGE` | 操作 | 是否使用 GPU |
| --- | --- | --- |
| `prepare` | 核对输入和输出计划 | 否 |
| `profile` | 采集 baseline/checkpoint 配对跟踪 | 是 |
| `phase` | 各重复三次阶段显存测量 | 是 |
| `close` | 复用清单中的证据并生成结论 | 否 |


```python
# ===== 学习者配置：本单元只声明本次运行意图 =====
RUN_LABEL = 'replace_with_unique_label'  # 默认占位符；写入正式结果前改为本次唯一标签。
RUN_STAGE = 'prepare'  # 默认 prepare 只读检查；profile=重采 trace；phase=重新测量；close=只读收口。

```

#### 5.2 固定实验配置与输入预检

本单元核对项目路径、固定证据清单、上游文件哈希和结果目录。`profile`、`phase` 还会检查训练依赖；`prepare`、`close` 不要求加载模型。


```python
"""加载固定证据清单并检查运行环境；不安装依赖、不加载模型、不创建输出目录。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

PROJECT_ID = '74'
SEMANTIC_ID = 'PERF-PROFILING-CLOSURE'
SCHEMA_VERSION = 'profiling-optimization-benchmark/v1'
EVIDENCE_MANIFEST_SCHEMA_VERSION = 'performance-evidence-manifest/v1'
RAW_PROFILE_SCHEMA_VERSION = 'profiling-trace-evidence/v1'
PHASE_MEMORY_SCHEMA_VERSION = 'training-phase-memory/v1'
EVIDENCE_MANIFEST_RELATIVE_PATH = Path(
    'benchmarks/results/74_evidence_manifest_rtx5070ti_profiling_closure_v5.json'
)
PRESSURE_VALIDATION_RELATIVE_PATH = None
PROFILE_STRATEGIES = ['baseline', 'checkpoint']
PHASE_REPEATS = 3
PHASE_PROCESS_MEMORY_FRACTION = 0.92
PHASE_MIN_FREE_MARGIN_MB = 512.0
REQUIRE_PHASE_MEMORY = True
ROOFLINE_COUNTER_SOURCE = 'not_collected'
QUALIFICATION_GATES = {
    'minimum_memory_saving_mb': 512.0,
    'minimum_throughput_ratio': 0.70,
    'maximum_loss_delta': 0.01,
}

if RUN_STAGE not in {'prepare', 'profile', 'phase', 'close'}:
    raise ValueError("RUN_STAGE 只能是 'prepare'、'profile'、'phase' 或 'close'。")
RUN_REAL_PROFILE = RUN_STAGE == 'profile'
RUN_PHASE_MEMORY = RUN_STAGE == 'phase'
RUN_REAL_PROJECT = RUN_STAGE == 'close'

GPU_PROJECT_ROOT = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
GPU_PROJECT_ROOT = next(
    (candidate for candidate in (GPU_PROJECT_ROOT, *GPU_PROJECT_ROOT.parents)
     if (candidate / 'tools/project_runtime.py').is_file()),
    GPU_PROJECT_ROOT,
)
if not (GPU_PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'tools/project_runtime.py').is_file():
        GPU_PROJECT_ROOT = colab_root
if not (GPU_PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    raise RuntimeError('找不到项目根目录：请设置 LLM_ALGO_PROJECT_ROOT，或先把仓库放到 /content/llm-algo-leetcode。')
os.chdir(GPU_PROJECT_ROOT)
if str(GPU_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_PROJECT_ROOT))

manifest_path = GPU_PROJECT_ROOT / EVIDENCE_MANIFEST_RELATIVE_PATH
if not manifest_path.is_file():
    raise FileNotFoundError(f'缺少固定证据清单：{manifest_path}')
evidence_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if (
    evidence_manifest.get('schema_version') != EVIDENCE_MANIFEST_SCHEMA_VERSION
    or evidence_manifest.get('semantic_id') != SEMANTIC_ID
):
    raise ValueError('固定证据清单的 schema 或 semantic_id 不匹配')

def check_manifest_entry(entry, name):
    if not isinstance(entry, dict) or not entry.get('path') or not entry.get('sha256'):
        raise ValueError(f'证据清单缺少 {name} 的 path/sha256')
    relative_path = Path(entry['path'])
    artifact_path = GPU_PROJECT_ROOT / relative_path
    if not artifact_path.is_file():
        raise FileNotFoundError(f'{name} 不存在：{artifact_path}')
    actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_sha256 != entry['sha256']:
        raise ValueError(f'{name} SHA256 不匹配：{relative_path}')
    return relative_path

manifest_sources = evidence_manifest.get('source_artifacts', {})
SOURCE_RESULTS = {
    name: check_manifest_entry(manifest_sources.get(name), f'{name} 上游结果')
    for name in ('baseline', 'strategy', 'budget')
}
ACTIVATION_SCOPE_RESULT_RELATIVE_PATH = check_manifest_entry(
    manifest_sources.get('activation_memory_scope'), 'activation_memory_scope 上游结果'
)
manifest_profile = evidence_manifest.get('profile', {})
check_manifest_entry(manifest_profile.get('report'), 'profiler 报告')
check_manifest_entry(manifest_profile.get('phase_memory', {}).get('report'), '阶段显存报告')

BASELINE_RESULT_RELATIVE_PATH = SOURCE_RESULTS['baseline']
STRATEGY_RESULT_RELATIVE_PATH = SOURCE_RESULTS['strategy']
BUDGET_RESULT_RELATIVE_PATH = SOURCE_RESULTS['budget']
PHASE_MEMORY_OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/{PROJECT_ID}_phase_memory_{RUN_LABEL}.json')
QUALIFICATION_OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/{PROJECT_ID}_profiling_optimization_{RUN_LABEL}.json')
output_parent = GPU_PROJECT_ROOT / QUALIFICATION_OUTPUT_RELATIVE_PATH.parent
output_parent_writable = output_parent.is_dir() and os.access(output_parent, os.W_OK)
requires_training_runtime = RUN_STAGE in {'profile', 'phase'}
transformers_available = importlib.util.find_spec('transformers') is not None
preflight_report = {
    'project_root': str(GPU_PROJECT_ROOT),
    'evidence_manifest': str(EVIDENCE_MANIFEST_RELATIVE_PATH),
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'sources': {name: str(path) for name, path in SOURCE_RESULTS.items()},
    'activation_scope': str(ACTIVATION_SCOPE_RESULT_RELATIVE_PATH),
    'output_parent': str(output_parent.relative_to(GPU_PROJECT_ROOT)),
    'output_parent_writable': output_parent_writable,
    'training_runtime_required': requires_training_runtime,
    'transformers_available': transformers_available,
    'ready': output_parent_writable and (transformers_available or not requires_training_runtime),
}
print(preflight_report)
if not preflight_report['ready']:
    raise RuntimeError('环境或输入资产预检未通过；没有执行实验。')

try:
    import torch
except ImportError:
    if requires_training_runtime:
        raise RuntimeError('当前阶段需要 PyTorch；请先安装与 CUDA 匹配的版本。')
    print('当前阶段不需要 PyTorch。')
else:
    print(f'torch: {torch.__version__}')
    print(f'cuda_build: {torch.version.cuda}; cuda_available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'gpu: {props.name}; memory_gib: {props.total_memory / 2**30:.1f}')

```

#### 5.3 输入合同与证据路径

从固定清单读取 73、75、76 的结果以及跟踪、阶段显存文件，核对模型、精度、批大小、随机种子和硬件。优化器与序列长度的差异作为两组实验的身份信息保留。

| 证据 | 用途 |
| --- | --- |
| 73/75/76 | 完整训练的基线、候选和预算门槛 |
| 配对跟踪 | 定位算子耗时与重算开销 |
| 阶段显存 | 比较前向、反向和优化器阶段峰值 |
| 76 长度阶梯 | 判断激活显存范围内的收益 |


```python
# 只读检查：自动继承固定 workload，并从已审核证据清单推导 trace/phase。
import hashlib
import json

if RUN_STAGE not in {'prepare', 'profile', 'phase', 'close'}:
    raise ValueError("RUN_STAGE 只能是 'prepare'、'profile'、'phase' 或 'close'。")
project_root = globals().get('GPU_PROJECT_ROOT', Path.cwd().resolve())
project_root = next(
    (candidate for candidate in (project_root, *project_root.parents) if (candidate / 'tools/project_runtime.py').is_file()),
    project_root,
)
if not (project_root / 'tools/project_runtime.py').is_file():
    raise RuntimeError(f'找不到项目根目录：{project_root}')
source_paths = {name: project_root / path for name, path in SOURCE_RESULTS.items()}
missing_sources = [str(path) for path in source_paths.values() if not path.is_file()]
activation_scope_path = project_root / ACTIVATION_SCOPE_RESULT_RELATIVE_PATH
evidence_manifest_path = project_root / EVIDENCE_MANIFEST_RELATIVE_PATH
for required_path in (activation_scope_path, evidence_manifest_path):
    if not required_path.is_file():
        missing_sources.append(str(required_path))
if missing_sources:
    raise FileNotFoundError('缺少上游结果：' + ', '.join(missing_sources))
baseline_source = json.loads(source_paths['baseline'].read_text(encoding='utf-8'))
strategy_source = json.loads(source_paths['strategy'].read_text(encoding='utf-8'))
budget_source = json.loads(source_paths['budget'].read_text(encoding='utf-8'))
activation_scope_source = json.loads(activation_scope_path.read_text(encoding='utf-8'))
evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding='utf-8'))
expected_semantic_ids = {
    'baseline': 'PERF-TRAINING-BASELINE',
    'strategy': 'PERF-MEMORY-STRATEGY',
    'budget': 'PERF-MEMORY-BUDGET',
}
actual_semantic_ids = {name: report.get('semantic_id') for name, report in {
    'baseline': baseline_source, 'strategy': strategy_source, 'budget': budget_source,
}.items()}
if actual_semantic_ids != expected_semantic_ids:
    raise ValueError(f'上游语义身份不匹配：{actual_semantic_ids}')
if evidence_manifest.get('schema_version') != EVIDENCE_MANIFEST_SCHEMA_VERSION or evidence_manifest.get('semantic_id') != SEMANTIC_ID:
    raise ValueError('证据清单的 schema 或 semantic_id 不匹配')
if activation_scope_source.get('schema_version') != 'training-performance-project/v1':
    raise ValueError('激活机制结果 schema 必须是 training-performance-project/v1')
if activation_scope_source.get('semantic_id') != 'PERF-MEMORY-STRATEGY':
    raise ValueError('激活机制结果 semantic_id 必须是 PERF-MEMORY-STRATEGY')
baseline_config = baseline_source.get('config', {})
strategy_config = strategy_source.get('config', {})
identity_fields = ('model_id', 'batch_size', 'seq_len', 'dtype', 'optimizer', 'workload', 'seed')
identity_mismatches = [key for key in identity_fields if baseline_config.get(key) != strategy_config.get(key)]
if identity_mismatches:
    raise ValueError(f'73/76 工作负载身份不一致：{identity_mismatches}')
activation_config = activation_scope_source.get('config', {})
activation_shared_fields = ('model_id', 'batch_size', 'dtype', 'seed', 'device')
activation_identity_mismatches = [
    key for key in activation_shared_fields
    if activation_config.get(key) != strategy_config.get(key)
]
if activation_identity_mismatches:
    raise ValueError(f'激活机制通道与完整训练通道身份不一致：{activation_identity_mismatches}')
if activation_config.get('optimizer') != 'SGD' or activation_config.get('optimizer_momentum') != 0.0:
    raise ValueError('激活机制通道必须使用无动量 SGD 隔离优化器状态')
if activation_config.get('comparison_scope') != 'activation_memory_isolation':
    raise ValueError('激活机制通道必须声明 comparison_scope=activation_memory_isolation')

def resolve_manifest_artifact(entry, name):
    if not isinstance(entry, dict) or not entry.get('path') or not entry.get('sha256'):
        raise ValueError(f'证据清单缺少 {name} 的 path/sha256')
    relative_path = Path(entry['path'])
    artifact_path = project_root / relative_path
    if not artifact_path.is_file():
        raise FileNotFoundError(f'{name} 不存在：{artifact_path}')
    actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_sha256 != entry['sha256']:
        raise ValueError(f'{name} SHA256 不匹配：{relative_path}')
    return relative_path


# 证据清单是唯一来源：路径与哈希一起推导，避免 TRACE/PHASE 标签漂移。
manifest_profile = evidence_manifest.get('profile', {})
manifest_profile_report = resolve_manifest_artifact(manifest_profile.get('report'), 'profiler 报告')
manifest_phase_report = resolve_manifest_artifact(
    manifest_profile.get('phase_memory', {}).get('report'), '阶段显存报告'
)
manifest_trace_dir = Path(manifest_profile.get('trace_dir', ''))
if not manifest_trace_dir or not (project_root / manifest_trace_dir).is_dir():
    raise FileNotFoundError(f'trace 目录不存在：{project_root / manifest_trace_dir}')

manifest_upstream = evidence_manifest.get('source_artifacts', {})
for name, local_path in source_paths.items():
    manifest_entry = manifest_upstream.get(name)
    resolved_path = resolve_manifest_artifact(manifest_entry, f'{name} 上游结果')
    if resolved_path != local_path.relative_to(project_root):
        raise ValueError(f'{name} 路径与证据清单不一致：{resolved_path}')

if RUN_STAGE == 'profile':
    PROFILE_TRACE_RELATIVE_DIR = Path(f'benchmarks/results/{PROJECT_ID}_profile/trace_{RUN_LABEL}')
    PROFILE_OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/{PROJECT_ID}_real_gpu_profile_{RUN_LABEL}.json')
else:
    PROFILE_TRACE_RELATIVE_DIR = manifest_trace_dir
    PROFILE_OUTPUT_RELATIVE_PATH = manifest_profile_report
PHASE_MEMORY_SOURCE_RELATIVE_PATH = (
    PHASE_MEMORY_OUTPUT_RELATIVE_PATH if RUN_STAGE == 'phase' else manifest_phase_report
)

PROFILE_MODEL_ID = strategy_config['model_id']
PROFILE_BATCH_SIZE = int(strategy_config['batch_size'])
PROFILE_SEQ_LEN = int(strategy_config['seq_len'])
PROFILE_WARMUP = int(strategy_config.get('warmup', 2))
PROFILE_ITERS = int(strategy_config.get('iters', 5))
if PROFILE_BATCH_SIZE <= 0 or PROFILE_SEQ_LEN <= 0 or PROFILE_WARMUP < 0 or PROFILE_ITERS <= 0:
    raise ValueError('上游 batch_size/seq_len/iters 必须 > 0，warmup 不能为负数。')
if not PROFILE_STRATEGIES or len(PROFILE_STRATEGIES) != len(set(PROFILE_STRATEGIES)):
    raise ValueError('PROFILE_STRATEGIES 不能为空且不能包含重复策略。')
if any(strategy not in {'baseline', 'checkpoint'} for strategy in PROFILE_STRATEGIES):
    raise ValueError('本节只采集 baseline/checkpoint；offload/hybrid 由上游预算门控处理。')
strategy_names = {item.get('name') for item in strategy_source.get('candidates', [])}
missing_candidates = [name for name in PROFILE_STRATEGIES if name not in strategy_names]
if missing_candidates:
    raise ValueError(f'76 节缺少配对候选：{missing_candidates}')
if SCHEMA_VERSION != 'profiling-optimization-benchmark/v1':
    raise ValueError('收口结果必须使用 profiling-optimization-benchmark/v1。')
PROFILE_IDENTITY = {
    'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE,
    'seq_len': PROFILE_SEQ_LEN, 'strategies': list(PROFILE_STRATEGIES),
    'warmup': PROFILE_WARMUP, 'iters': PROFILE_ITERS, 'phase_repeats': PHASE_REPEATS,
    'run_label': RUN_LABEL, 'run_stage': RUN_STAGE,
    'evidence_manifest': str(EVIDENCE_MANIFEST_RELATIVE_PATH),
    'adoption_metrics': 'unprofiled_repeated_measurements',
    'profiler_metrics': 'bottleneck_attribution_only',
    'pressure_validation': str(PRESSURE_VALIDATION_RELATIVE_PATH) if PRESSURE_VALIDATION_RELATIVE_PATH else None,
    'decision_scopes': {
        'full_training_adoption': {'optimizer': strategy_config.get('optimizer'), 'seq_len': strategy_config.get('seq_len')},
        'activation_memory_isolation': {
            'optimizer': activation_config.get('optimizer'),
            'tested_seq_lens': activation_scope_source.get('summary', {}).get('tested_seq_lens'),
        },
    },
    'resolved_profile_report': str(PROFILE_OUTPUT_RELATIVE_PATH),
    'resolved_trace_dir': str(PROFILE_TRACE_RELATIVE_DIR),
    'resolved_phase_report': str(PHASE_MEMORY_SOURCE_RELATIVE_PATH),
    'activation_scope_source': str(ACTIVATION_SCOPE_RESULT_RELATIVE_PATH),
    'output_exists': {
        'profile': (project_root / PROFILE_OUTPUT_RELATIVE_PATH).is_file(),
        'phase': (project_root / PHASE_MEMORY_SOURCE_RELATIVE_PATH).is_file(),
        'closure': (project_root / QUALIFICATION_OUTPUT_RELATIVE_PATH).is_file(),
    },
}
print(PROFILE_IDENTITY)

```

#### 5.4 采集专项证据

`profile` 采集配对跟踪，`phase` 采集阶段显存；`prepare` 和 `close` 跳过 GPU。完整训练的采用门槛仍使用无分析器的重复测量，跟踪和阶段数据只用于解释瓶颈。


```python
# 按配置采集原始 trace 和 JSON；本 cell 不修改判定规则。
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

if RUN_REAL_PROFILE:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from tools.model_runtime import resolve_model
    # Jupyter 可能缓存旧版 helper；显式 reload，避免修改工具文件后必须重启内核。
    import importlib
    import tools.profiling_runtime as profiling_runtime
    profiling_runtime = importlib.reload(profiling_runtime)
    collect_training_trace = profiling_runtime.collect_training_trace
    from tools.project_runtime import environment_preflight, save_json_record_once, standard_experiment_config, standard_training_metrics
    preflight = environment_preflight(torch, required_packages=('transformers',), require_gpu=True)
    print({'environment_preflight': preflight})
    if not preflight['ready']:
        raise RuntimeError('环境预检未通过，请先按 next_actions 修复；没有开始 profiling。')

    profile_output_path = project_root / PROFILE_OUTPUT_RELATIVE_PATH
    trace_root = project_root / PROFILE_TRACE_RELATIVE_DIR
    if profile_output_path.exists() or trace_root.exists():
        raise FileExistsError(f'profiler 结果已存在，请更换 RUN_LABEL：{RUN_LABEL}')

    baseline_result_path = project_root / BASELINE_RESULT_RELATIVE_PATH
    if not baseline_result_path.exists():
        raise FileNotFoundError(f'找不到上游训练基线：{baseline_result_path}')
    baseline_result = json.loads(baseline_result_path.read_text(encoding='utf-8'))
    baseline_result_config = baseline_result.get('config', {})
    strategy_result_path = project_root / STRATEGY_RESULT_RELATIVE_PATH
    if not strategy_result_path.exists():
        raise FileNotFoundError(f'找不到上游策略结果：{strategy_result_path}')
    strategy_result = json.loads(strategy_result_path.read_text(encoding='utf-8'))
    strategy_result_config = strategy_result.get('config', {})
    missing_repeats = [item.get('name', '<unknown>') for item in strategy_result.get('candidates', []) if len(item.get('runs', [])) < 3]
    if missing_repeats:
        raise ValueError(f'策略结果必须先完成每个候选的 3 次重复运行：{missing_repeats}')
    for key, value in {'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN}.items():
        if baseline_result_config.get(key) != value:
            raise ValueError(f'当前 workload 与上游训练基线不一致：{key}={baseline_result_config.get(key)} != {value}')
        if strategy_result_config.get(key) != value:
            raise ValueError(f'当前 workload 与上游策略结果不一致：{key}={strategy_result_config.get(key)} != {value}')

    model_path = resolve_model(PROFILE_MODEL_ID, source='auto', cache_dir='model_cache')
    model_config = AutoConfig.from_pretrained(model_path)
    generator = torch.Generator(device='cpu').manual_seed(42)
    input_ids_cpu = torch.randint(0, model_config.vocab_size, (PROFILE_BATCH_SIZE, PROFILE_SEQ_LEN), generator=generator)
    trace_root.mkdir(parents=True, exist_ok=False)
    candidates = []

    for strategy in PROFILE_STRATEGIES:
        trace_dir = trace_root / strategy
        trace_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        model.config.use_cache = False
        if strategy == 'checkpoint':
            model.gradient_checkpointing_enable()
        model.to('cuda').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        input_ids = input_ids_cpu.to('cuda')
        labels = input_ids.clone()

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function('phase.forward'):
                loss = model(input_ids=input_ids, labels=labels).loss
            with torch.profiler.record_function('phase.backward'):
                loss.backward()
            with torch.profiler.record_function('phase.optimizer'):
                optimizer.step()
            return loss

        measured = collect_training_trace(
            train_step, torch_module=torch, output_dir=trace_dir,
            warmup=PROFILE_WARMUP, iters=PROFILE_ITERS,
            batch_size=PROFILE_BATCH_SIZE,
        )
        trace_files = sorted(trace_dir.glob('*.pt.trace.json'))
        if not trace_files:
            raise RuntimeError(f'{strategy} 未生成 .pt.trace.json；请检查 profiler schedule 和输出目录：{trace_dir}')
        peak_allocated_mb = measured['peak_memory_mb']
        peak_reserved_mb = measured['peak_reserved_mb']
        # 向后兼容旧 helper 返回值；新 helper 会直接提供 allocator_slack_mb。
        allocator_slack_mb = measured.get(
            'allocator_slack_mb', round(peak_reserved_mb - peak_allocated_mb, 2)
        )
        candidates.append({
            'name': strategy, 'status': 'ok', 'step_time_ms': measured['step_time_ms'],
            'samples_per_s': measured['samples_per_s'], 'loss': measured['loss'],
            'peak_memory_mb': peak_allocated_mb, 'peak_reserved_mb': peak_reserved_mb,
            'allocator_slack_mb': allocator_slack_mb,
            'trace_dir': str(trace_dir.relative_to(project_root)), 'top_operators': measured['top_operators'],
        })
        del optimizer, model, input_ids, labels
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        'schema_version': RAW_PROFILE_SCHEMA_VERSION, 'semantic_id': SEMANTIC_ID, 'run_mode': 'real_gpu', 'run_label': RUN_LABEL,
        'task': 'profiling_driven_optimization', 'stage': 'real_profiler_trace',
        'config': {'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN, 'dtype': 'float32', 'strategies': PROFILE_STRATEGIES, 'warmup': PROFILE_WARMUP, 'iters': PROFILE_ITERS},
        'sources': {'baseline': str(baseline_result_path.relative_to(project_root)), 'strategy': str(strategy_result_path.relative_to(project_root))}, 'trace_root': str(trace_root.relative_to(project_root)),
        'candidates': candidates, 'profile': {'tool': 'torch.profiler', 'activities': ['CPU', 'CUDA'], 'status': 'collected', 'measurement_role': 'bottleneck_attribution_only', 'timing_for_adoption': False, 'active_steps': max(1, PROFILE_ITERS - 1), 'repeat': 1, 'trace_files_by_strategy': {item['name']: [str(path.relative_to(project_root)) for path in sorted((trace_root / item['name']).glob('*.pt.trace.json'))] for item in candidates}},
    }
    report['experiment'] = standard_experiment_config({
        'model_id': PROFILE_MODEL_ID, 'backend': 'torch.profiler',
        'dtype': 'float32', 'optimizer': 'AdamW',
        'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN,
        'warmup': PROFILE_WARMUP, 'iters': PROFILE_ITERS, 'seed': 42,
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
    })
    report['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    output_path = project_root / PROFILE_OUTPUT_RELATIVE_PATH
    save_json_record_once(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
elif not RUN_PHASE_MEMORY:
    print(f'跳过 GPU 归因实验：当前 RUN_STAGE={RUN_STAGE!r}。')

if RUN_PHASE_MEMORY:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from tools.model_runtime import resolve_model
    from tools.project_runtime import environment_preflight, save_json_record_once, standard_experiment_config, standard_training_metrics

    import importlib
    import tools.training_memory_runtime as training_memory_runtime
    import tools.profiling_runtime as profiling_runtime
    import tools.profiling_result_schema as profiling_result_schema
    training_memory_runtime = importlib.reload(training_memory_runtime)
    profiling_runtime = importlib.reload(profiling_runtime)
    profiling_result_schema = importlib.reload(profiling_result_schema)
    measure_phased_training_run = training_memory_runtime.measure_phased_training_run
    summarize_phase_memory_runs = profiling_runtime.summarize_phase_memory_runs
    validate_phase_memory_result = profiling_result_schema.validate_phase_memory_result

    preflight = environment_preflight(torch, required_packages=('transformers',), require_gpu=True)
    print({'environment_preflight': preflight})
    if not preflight['ready']:
        raise RuntimeError('环境预检未通过；没有开始阶段显存实验。')
    if not 0 < PHASE_PROCESS_MEMORY_FRACTION <= 1:
        raise ValueError('PHASE_PROCESS_MEMORY_FRACTION 必须位于 (0, 1]。')
    torch.cuda.set_per_process_memory_fraction(PHASE_PROCESS_MEMORY_FRACTION, 0)

    phase_output_path = project_root / PHASE_MEMORY_OUTPUT_RELATIVE_PATH
    if phase_output_path.exists():
        raise FileExistsError(f'阶段显存结果已存在，请更换 RUN_LABEL：{RUN_LABEL}')

    baseline_result_path = project_root / BASELINE_RESULT_RELATIVE_PATH
    strategy_result_path = project_root / STRATEGY_RESULT_RELATIVE_PATH
    baseline_result = json.loads(baseline_result_path.read_text(encoding='utf-8'))
    strategy_result = json.loads(strategy_result_path.read_text(encoding='utf-8'))
    for key, value in {'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN}.items():
        if baseline_result.get('config', {}).get(key) != value or strategy_result.get('config', {}).get(key) != value:
            raise ValueError(f'阶段显存 workload 与 73/76 不一致：{key}')

    source_candidates = {item.get('name'): item for item in strategy_result.get('candidates', [])}
    projected_peak_reserved_mb = source_candidates.get('baseline', {}).get('peak_reserved_mb')
    if not isinstance(projected_peak_reserved_mb, (int, float)) or projected_peak_reserved_mb <= 0:
        raise ValueError('76 节缺少 baseline.peak_reserved_mb，无法执行显存安全准入。')
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_memory_mb = free_bytes / 1024**2
    total_memory_mb = total_bytes / 1024**2
    required_free_memory_mb = projected_peak_reserved_mb + PHASE_MIN_FREE_MARGIN_MB
    phase_safety = {
        'free_memory_mb_before_load': round(free_memory_mb, 2),
        'total_memory_mb': round(total_memory_mb, 2),
        'projected_peak_reserved_mb': round(projected_peak_reserved_mb, 2),
        'minimum_free_margin_mb': PHASE_MIN_FREE_MARGIN_MB,
        'required_free_memory_mb': round(required_free_memory_mb, 2),
        'admitted': free_memory_mb >= required_free_memory_mb,
    }
    print({'phase_memory_safety': phase_safety})
    if not phase_safety['admitted']:
        raise RuntimeError('GPU 空闲显存不足；关闭其他 GPU 进程后重试，阶段实验尚未开始。')

    model_path = resolve_model(PROFILE_MODEL_ID, source='auto', cache_dir='model_cache')
    model_config = AutoConfig.from_pretrained(model_path)
    generator = torch.Generator(device='cpu').manual_seed(42)
    input_ids_cpu = torch.randint(0, model_config.vocab_size, (PROFILE_BATCH_SIZE, PROFILE_SEQ_LEN), generator=generator)
    phase_candidates = []

    for strategy in PROFILE_STRATEGIES:
        runs = []
        for repeat_index in range(PHASE_REPEATS):
            model = optimizer = input_ids = labels = None
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
                model.config.use_cache = False
                if strategy == 'checkpoint':
                    model.gradient_checkpointing_enable()
                model.to('cuda').train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                input_ids = input_ids_cpu.to('cuda')
                labels = input_ids.clone()

                def phase_train_step(record_phase):
                    optimizer.zero_grad(set_to_none=True)
                    loss = record_phase(
                        'forward', lambda: model(input_ids=input_ids, labels=labels).loss,
                    )
                    record_phase('backward', loss.backward)
                    record_phase('optimizer', optimizer.step)
                    return loss

                measured = measure_phased_training_run(
                    phase_train_step, torch_module=torch,
                    batch_size=PROFILE_BATCH_SIZE, warmup=PROFILE_WARMUP, iters=PROFILE_ITERS,
                )
                runs.append({'repeat': repeat_index + 1, 'status': 'ok', **measured})
            except torch.OutOfMemoryError as exc:
                runs.append({'repeat': repeat_index + 1, 'status': 'oom', 'error': str(exc)})
            finally:
                del optimizer, model, input_ids, labels
                gc.collect()
                torch.cuda.empty_cache()
        phase_candidates.append(summarize_phase_memory_runs(strategy, runs))

    phase_by_name = {item['name']: item for item in phase_candidates}
    phase_summary = {'candidate_count': len(phase_candidates), 'successful_candidates': [item['name'] for item in phase_candidates if item['status'] == 'ok']}
    if all(phase_by_name.get(name, {}).get('status') == 'ok' for name in PROFILE_STRATEGIES):
        phase_summary['allocated_memory_saving_mb'] = {
            phase: round(phase_by_name['baseline']['phase_metrics'][phase]['peak_allocated_mb'] - phase_by_name['checkpoint']['phase_metrics'][phase]['peak_allocated_mb'], 3)
            for phase in ('forward', 'backward', 'optimizer')
        }
        phase_summary['global_peak_phase'] = {name: phase_by_name[name]['global_peak_phase'] for name in PROFILE_STRATEGIES}

    phase_report = {
        'schema_version': PHASE_MEMORY_SCHEMA_VERSION, 'semantic_id': SEMANTIC_ID,
        'run_mode': 'real_gpu', 'run_label': RUN_LABEL, 'task': 'profiling_driven_optimization',
        'stage': 'phase_memory_attribution', 'evidence_level': 'repeated_phase_memory_measurement',
        'config': {
            'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN,
            'dtype': 'float32', 'optimizer': 'AdamW', 'strategies': PROFILE_STRATEGIES,
            'warmup': PROFILE_WARMUP, 'iters': PROFILE_ITERS, 'repeats': PHASE_REPEATS,
            'seed': 42, 'process_memory_fraction': PHASE_PROCESS_MEMORY_FRACTION,
            'minimum_free_margin_mb': PHASE_MIN_FREE_MARGIN_MB,
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
        },
        'sources': {
            'baseline': str(baseline_result_path.relative_to(project_root)),
            'strategy': str(strategy_result_path.relative_to(project_root)),
            'profile_trace': str(PROFILE_OUTPUT_RELATIVE_PATH),
        },
        'measurement': {'role': 'phase_memory_attribution_only', 'timing_for_adoption': False, 'synchronize_at_phase_boundaries': True},
        'safety': phase_safety,
        'candidates': phase_candidates, 'summary': phase_summary,
    }
    phase_report['experiment'] = standard_experiment_config({
        **phase_report['config'], 'backend': 'torch', 'workload': strategy_result.get('config', {}).get('workload'),
        'cuda': torch.version.cuda,
    })
    phase_report['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in phase_candidates}
    phase_errors = validate_phase_memory_result(phase_report)
    if phase_errors:
        raise ValueError('阶段显存结果契约校验失败：' + '; '.join(phase_errors))
    save_json_record_once(phase_output_path, phase_report)
    print(json.dumps(phase_report, ensure_ascii=False, indent=2))

```

#### 5.5 核对证据

本单元读取并核对两组实验。`prepare` 展示证据计划；`profile`、`phase` 检查本次新增文件；`close` 使用固定清单中的完整证据。完整训练和激活显存分别计算，不互相替代。


```python
import hashlib
import importlib
import json
from pathlib import Path
import sys

project_root = Path.cwd().resolve()
for candidate_root in (project_root, *project_root.parents):
    if (candidate_root / 'tools/project_runtime.py').is_file():
        if str(candidate_root) not in sys.path:
            sys.path.insert(0, str(candidate_root))
        break
from tools.project_runtime import resolve_project_root, save_json_record_once, standard_experiment_config, standard_training_metrics
from tools.profiling_result_schema import validate_phase_memory_result, validate_result as validate_profiling_result
import tools.profiling_runtime as profiling_runtime_module
profiling_runtime_module = importlib.reload(profiling_runtime_module)
infer_bottleneck_assessments = profiling_runtime_module.infer_bottleneck_assessments
import tools.performance_handoff_runtime as performance_handoff_runtime_module
performance_handoff_runtime_module = importlib.reload(performance_handoff_runtime_module)
make_performance_handoff = performance_handoff_runtime_module.build_performance_handoff
build_workload_identity = performance_handoff_runtime_module.build_workload_identity
qualify_memory_candidates = performance_handoff_runtime_module.qualify_memory_candidates

PROJECT_ROOT = resolve_project_root()
UPSTREAM_PATHS = {
    'baseline': PROJECT_ROOT / BASELINE_RESULT_RELATIVE_PATH,
    'strategy': PROJECT_ROOT / STRATEGY_RESULT_RELATIVE_PATH,
    'budget': PROJECT_ROOT / BUDGET_RESULT_RELATIVE_PATH,
}
OUTPUT_PATH = PROJECT_ROOT / QUALIFICATION_OUTPUT_RELATIVE_PATH
HANDOFF_PATH = PROJECT_ROOT / f'benchmarks/results/{PROJECT_ID}_profile/performance_handoff_{RUN_LABEL}.json'
PROFILE_TRACE_DIR = PROJECT_ROOT / PROFILE_TRACE_RELATIVE_DIR
PROFILE_REPORT_PATH = PROJECT_ROOT / PROFILE_OUTPUT_RELATIVE_PATH
PHASE_MEMORY_REPORT_PATH = PROJECT_ROOT / PHASE_MEMORY_SOURCE_RELATIVE_PATH
ACTIVATION_SCOPE_REPORT_PATH = PROJECT_ROOT / ACTIVATION_SCOPE_RESULT_RELATIVE_PATH
PRESSURE_VALIDATION_PATH = (
    PROJECT_ROOT / PRESSURE_VALIDATION_RELATIVE_PATH
    if PRESSURE_VALIDATION_RELATIVE_PATH is not None else None
)

def load_json_object(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'结果必须是 JSON object：{path}')
    return payload

def load_required_reports(paths):
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('缺少上游报告：' + ', '.join(missing))
    return {name: load_json_object(path) for name, path in paths.items()}

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def relative_artifact(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(PROJECT_ROOT)), 'sha256': sha256_file(path)}

def inspect_profile_trace(trace_dir, expected_strategies=None):
    """只确认 trace 是否存在，不把文件存在误写成瓶颈结论。"""
    trace_dir = Path(trace_dir)
    expected_strategies = list(expected_strategies or [])
    trace_files = sorted(path for path in trace_dir.rglob('*.pt.trace.json')) if trace_dir.is_dir() else []
    strategy_dirs = {name: sorted((trace_dir / name).glob('*.pt.trace.json')) for name in expected_strategies}
    missing_strategies = [name for name, files in strategy_dirs.items() if not files]
    return {
        'status': 'collected' if trace_files else 'not_collected',
        'tool': 'torch.profiler' if trace_files else None,
        'trace_dir': str(trace_dir.relative_to(PROJECT_ROOT)) if trace_dir.is_relative_to(PROJECT_ROOT) else str(trace_dir),
        'trace_files': [str(path.relative_to(PROJECT_ROOT)) for path in trace_files],
        'expected_strategies': expected_strategies,
        'missing_strategies': missing_strategies,
        'trace_files_by_strategy': {
            name: [str(path.relative_to(PROJECT_ROOT)) for path in files]
            for name, files in strategy_dirs.items()
        },
        'measurement_role': 'bottleneck_attribution_only',
        'timing_for_adoption': False,
    }

def load_pressure_validation(base_config):
    if PRESSURE_VALIDATION_PATH is None:
        return {'status': 'not_collected', 'reason': 'optional_pressure_validation_path_not_set'}
    if not PRESSURE_VALIDATION_PATH.is_file():
        raise FileNotFoundError(f'压力验证结果不存在：{PRESSURE_VALIDATION_PATH}')
    report = load_json_object(PRESSURE_VALIDATION_PATH)
    config = report.get('config', {})
    fixed = ('model_id', 'batch_size', 'dtype', 'optimizer')
    mismatches = [key for key in fixed if config.get(key) != base_config.get(key)]
    if mismatches:
        raise ValueError(f'压力验证与 G0 身份不一致：{mismatches}')
    if not isinstance(config.get('seq_len'), int) or config['seq_len'] <= base_config.get('seq_len', 0):
        raise ValueError('压力验证 seq_len 必须大于 G0 seq_len')
    return {'status': 'collected', 'artifact': relative_artifact(PRESSURE_VALIDATION_PATH), 'config': config}

def assess_activation_memory_scope(report, full_training_config):
    """独立重算激活隔离通道，绝不把 SGD 结论提升为 AdamW 采用结论。"""
    config = report.get('config', {})
    if report.get('schema_version') != 'training-performance-project/v1':
        raise ValueError('激活机制结果 schema 不匹配')
    if report.get('semantic_id') != 'PERF-MEMORY-STRATEGY':
        raise ValueError('激活机制结果 semantic_id 不匹配')
    if config.get('comparison_scope') != 'activation_memory_isolation':
        raise ValueError('激活机制结果缺少 activation_memory_isolation 范围声明')
    shared_fields = ('model_id', 'batch_size', 'dtype', 'seed', 'device')
    mismatches = [key for key in shared_fields if config.get(key) != full_training_config.get(key)]
    if mismatches:
        raise ValueError(f'激活机制通道共享身份不匹配：{mismatches}')
    if config.get('optimizer') != 'SGD' or config.get('optimizer_momentum') != 0.0:
        raise ValueError('激活机制通道必须使用无动量 SGD')

    comparisons = {item.get('name'): item for item in report.get('summary', {}).get('comparisons', [])}
    comparison = comparisons.get('checkpoint')
    if not comparison:
        raise ValueError('激活机制结果缺少 checkpoint comparison')
    comparison_seq_len = comparison.get('comparison_seq_len')
    length_record = next(
        (item for item in report.get('length_records', []) if item.get('seq_len') == comparison_seq_len),
        None,
    )
    if not length_record:
        raise ValueError(f'激活机制结果缺少 seq_len={comparison_seq_len} 的原始记录')
    candidates = {item.get('name'): item for item in length_record.get('candidates', [])}
    baseline = candidates.get('baseline')
    checkpoint = candidates.get('checkpoint')
    if not baseline or not checkpoint:
        raise ValueError('激活机制原始记录必须包含 baseline/checkpoint')
    if any(item.get('status') != 'ok' or item.get('successful_runs', 0) < 3 for item in (baseline, checkpoint)):
        raise ValueError('激活机制 baseline/checkpoint 必须各有至少 3 次成功重复')

    memory_saving_mb = baseline['peak_memory_mb'] - checkpoint['peak_memory_mb']
    throughput_ratio = checkpoint['samples_per_s'] / baseline['samples_per_s']
    loss_delta = abs(
        checkpoint.get('eval_loss', checkpoint.get('loss'))
        - baseline.get('eval_loss', baseline.get('loss'))
    )
    passed = (
        memory_saving_mb >= QUALIFICATION_GATES['minimum_memory_saving_mb']
        and throughput_ratio >= QUALIFICATION_GATES['minimum_throughput_ratio']
        and loss_delta <= QUALIFICATION_GATES['maximum_loss_delta']
        and comparison.get('accepted') is True
    )
    return {
        'scope': 'activation_memory_isolation',
        'decision': 'accept_for_activation_memory_scope' if passed else 'tune',
        'reason': (
            'checkpoint_passed_activation_memory_and_throughput_gates'
            if passed else 'checkpoint_failed_activation_scope_gates'
        ),
        'model_id': config.get('model_id'),
        'optimizer': 'SGD',
        'optimizer_momentum': 0.0,
        'comparison_seq_len': comparison_seq_len,
        'metrics': {
            'memory_saving_mb': round(memory_saving_mb, 3),
            'throughput_ratio': round(throughput_ratio, 6),
            'loss_delta': round(loss_delta, 8),
            'baseline_peak_memory_mb': baseline['peak_memory_mb'],
            'checkpoint_peak_memory_mb': checkpoint['peak_memory_mb'],
        },
        'gates': dict(QUALIFICATION_GATES),
        'evidence': relative_artifact(ACTIVATION_SCOPE_REPORT_PATH),
        'evidence_status': report.get('status'),
        'tested_seq_lens': report.get('summary', {}).get('tested_seq_lens', []),
        'promotable_to_full_training': False,
        'limitations': [
            'optimizer_state_is_intentionally_removed_with_zero_momentum_sgd',
            'result_does_not_establish_adamw_full_training_adoption',
            'length_ladder_stopped_at_the_safety_boundary',
        ],
    }


def build_upstream_report(reports):
    if not PROFILE_REPORT_PATH.is_file():
        raise FileNotFoundError(f'缺少 profiler 配对报告：{PROFILE_REPORT_PATH}')
    profile_report = load_json_object(PROFILE_REPORT_PATH)
    if profile_report.get('schema_version') != RAW_PROFILE_SCHEMA_VERSION:
        raise ValueError(f'trace 报告 schema 不匹配：{profile_report.get("schema_version")}')
    phase_report = None
    if PHASE_MEMORY_REPORT_PATH.is_file():
        phase_report = load_json_object(PHASE_MEMORY_REPORT_PATH)
        if phase_report.get('schema_version') != PHASE_MEMORY_SCHEMA_VERSION:
            raise ValueError(f'阶段显存报告 schema 不匹配：{phase_report.get("schema_version")}')
        phase_errors = validate_phase_memory_result(phase_report)
        if phase_errors:
            raise ValueError('阶段显存结果契约校验失败：' + '; '.join(phase_errors))
        phase_status = {item.get('name'): item.get('status') for item in phase_report.get('candidates', [])}
        if REQUIRE_PHASE_MEMORY and any(phase_status.get(name) != 'ok' for name in PROFILE_STRATEGIES):
            raise ValueError(f'阶段显存报告未完成全部配对策略：{phase_status}')
    elif REQUIRE_PHASE_MEMORY:
        raise FileNotFoundError(f'缺少阶段显存报告：{PHASE_MEMORY_REPORT_PATH}')
    expected_strategies = [item.get('name') for item in profile_report.get('candidates', []) if item.get('name')]
    profile = inspect_profile_trace(PROFILE_TRACE_DIR, expected_strategies=expected_strategies)
    profile['report'] = relative_artifact(PROFILE_REPORT_PATH)
    profile['top_operators'] = {item['name']: item.get('top_operators', '') for item in profile_report.get('candidates', [])}
    profile['phase_memory'] = (
        {'status': 'collected', 'report': relative_artifact(PHASE_MEMORY_REPORT_PATH), 'timing_for_adoption': False}
        if phase_report is not None else {'status': 'not_collected'}
    )
    workload_identity = build_workload_identity(reports['baseline'], reports['strategy'])
    bottleneck_assessments = infer_bottleneck_assessments(
        profile_report, evidence_path=str(PROFILE_OUTPUT_RELATIVE_PATH),
        phase_report=phase_report,
        phase_evidence_path=str(PHASE_MEMORY_SOURCE_RELATIVE_PATH) if phase_report else None,
    )
    qualification = qualify_memory_candidates(
        reports['strategy'], reports['budget'], profile_report,
        bottleneck_assessments, QUALIFICATION_GATES,
    )
    if qualification['qualified_candidates']:
        decision = {'decision': 'accept', 'reason': 'at_least_one_candidate_passed_profiling_qualification', 'next_action': 'write_part05_handoff'}
    elif any(item['decision'] == 'tune' for item in qualification['candidate_records']):
        tune_records = [item for item in qualification['candidate_records'] if item['decision'] == 'tune']
        only_memory_gate = all(item.get('failed_gates') == ['memory_saving'] and not item.get('missing_evidence') for item in tune_records)
        decision = {
            'decision': 'tune',
            'reason': 'global_peak_memory_saving_below_threshold' if only_memory_gate else 'candidate_requires_more_evidence_or_threshold_tuning',
            'next_action': 'retain_phase_evidence_and_do_not_adopt_at_fixed_workload' if phase_report else 'collect_phase_memory_evidence',
        }
    else:
        decision = {'decision': 'reject', 'reason': 'all_candidates_failed_hard_gates', 'next_action': 'restore_baseline'}
    activation_scope_report = load_json_object(ACTIVATION_SCOPE_REPORT_PATH)
    activation_scope_decision = assess_activation_memory_scope(
        activation_scope_report, reports['strategy'].get('config', {})
    )
    full_training_scope_decision = {
        'scope': 'full_training_adoption',
        **decision,
        'optimizer': reports['strategy'].get('config', {}).get('optimizer'),
        'seq_len': reports['strategy'].get('config', {}).get('seq_len'),
    }
    decision_scopes = {
        'full_training_adoption': full_training_scope_decision,
        'activation_memory_mechanism': activation_scope_decision,
    }
    source_artifacts = {name: relative_artifact(path) for name, path in UPSTREAM_PATHS.items()}
    source_artifacts['activation_memory_scope'] = relative_artifact(ACTIVATION_SCOPE_REPORT_PATH)
    if phase_report is not None:
        source_artifacts['phase_memory'] = relative_artifact(PHASE_MEMORY_REPORT_PATH)
    report = {
        'schema_version': SCHEMA_VERSION,
        'semantic_id': SEMANTIC_ID,
        'run_label': RUN_LABEL,
        'project': PROJECT_ID,
        'stage': 'profiling_driven_optimization',
        'config': standard_experiment_config(reports['strategy'].get('config', {})),
        'candidates': qualification['candidate_records'],
        'profile': profile,
        'bottleneck': bottleneck_assessments,
        'validation': {
            'workload_identity': workload_identity,
            'pressure_validation': load_pressure_validation(reports['strategy'].get('config', {})),
            'source_artifacts': source_artifacts,
            'evidence_roles': {
                'adoption_metrics': 'unprofiled_repeated_measurements',
                'profiler_metrics': 'bottleneck_attribution_only',
                'activation_scope_metrics': 'scope_limited_unprofiled_repeated_measurements',
            },
            'scope_boundary': {
                'top_level_decision_scope': 'full_training_adoption',
                'activation_scope_promotable_to_full_training': False,
            },
        },
        'decision': decision,
        'decision_scopes': decision_scopes,
        'standard_metrics': {
            item.get('name', f'candidate_{index}'): standard_training_metrics(item)
            for index, item in enumerate(reports['strategy'].get('candidates', []))
            if isinstance(item, dict)
        },
    }
    errors = validate_profiling_result(report)
    if errors:
        raise ValueError('专项结果契约校验失败：' + '; '.join(errors))
    return report, qualification, source_artifacts, workload_identity, decision_scopes

upstream_reports = load_required_reports(UPSTREAM_PATHS)
profile_report_read = load_json_object(PROFILE_REPORT_PATH) if PROFILE_REPORT_PATH.is_file() else None
phase_report_read = load_json_object(PHASE_MEMORY_REPORT_PATH) if PHASE_MEMORY_REPORT_PATH.is_file() else None
activation_scope_report_read = load_json_object(ACTIVATION_SCOPE_REPORT_PATH) if ACTIVATION_SCOPE_REPORT_PATH.is_file() else None
activation_scope_review = (
    assess_activation_memory_scope(activation_scope_report_read, upstream_reports['strategy'].get('config', {}))
    if activation_scope_report_read else None
)
profile_schema_ok = bool(profile_report_read and profile_report_read.get('schema_version') == RAW_PROFILE_SCHEMA_VERSION)
phase_errors = validate_phase_memory_result(phase_report_read) if phase_report_read else ['phase report missing']
evidence_review = {
    'workload_identity': build_workload_identity(upstream_reports['baseline'], upstream_reports['strategy']),
    'profile': {
        'path': str(PROFILE_OUTPUT_RELATIVE_PATH),
        'schema_ok': profile_schema_ok,
        'evidence_level': profile_report_read.get('profile', {}).get('measurement_role') if profile_report_read else None,
    },
    'phase_memory': {
        'path': str(PHASE_MEMORY_SOURCE_RELATIVE_PATH),
        'schema_ok': not phase_errors,
        'errors': phase_errors,
        'evidence_level': phase_report_read.get('evidence_level') if phase_report_read else None,
    },
    'activation_memory_scope': {
        'path': str(ACTIVATION_SCOPE_RESULT_RELATIVE_PATH),
        'decision': activation_scope_review.get('decision') if activation_scope_review else None,
        'promotable_to_full_training': False,
    },
    'ready_for_decision': (
        profile_schema_ok
        and (not REQUIRE_PHASE_MEMORY or not phase_errors)
        and activation_scope_review is not None
    ),
}
print(json.dumps(evidence_review, ensure_ascii=False, indent=2))

# 运行本单元，自动整理真实 profiler 数据和结论；不把 profiler 时间用于采用门控。
import json
from pathlib import Path

project_root = Path.cwd().resolve()
for candidate_root in (project_root, *project_root.parents):
    if (candidate_root / 'benchmarks').is_dir() and (candidate_root / '02_PyTorch_Algorithms').is_dir():
        project_root = candidate_root
        break
PROJECT_ROOT = project_root
profile_report_path = PROJECT_ROOT / PROFILE_OUTPUT_RELATIVE_PATH
phase_report_path = PROJECT_ROOT / PHASE_MEMORY_SOURCE_RELATIVE_PATH
close_report_path = PROJECT_ROOT / QUALIFICATION_OUTPUT_RELATIVE_PATH
if not profile_report_path.is_file():
    print(f'尚未生成 profiler 报告：{profile_report_path}')
    print(f'当前 profiler 来源由证据清单推导：{EVIDENCE_MANIFEST_RELATIVE_PATH}')
    profile_report = {}
else:
    profile_report = json.loads(profile_report_path.read_text(encoding='utf-8'))
rows = [{
    'strategy': item.get('name'),
    'profile_step_time_ms': item.get('step_time_ms'),
    'profile_samples_per_s': item.get('samples_per_s'),
    'peak_allocated_mb': item.get('peak_memory_mb'),
    'peak_reserved_mb': item.get('peak_reserved_mb'),
    'allocator_slack_mb': item.get('allocator_slack_mb'),
    'status': item.get('status'),
    'trace_dir': item.get('trace_dir'),
} for item in profile_report.get('candidates', [])]

if profile_report:
    print('Profiler 归因数据（不用于采用门控）')
    print(json.dumps({'config': profile_report.get('config'), 'candidates': rows, 'trace': profile_report.get('profile')}, ensure_ascii=False, indent=2))
if len(rows) == 2 and all(row['status'] == 'ok' for row in rows):
    base = next(row for row in rows if row['strategy'] == 'baseline')
    tuned = next(row for row in rows if row['strategy'] == 'checkpoint')
    print(f"allocated 变化：{base['peak_allocated_mb']:.2f} → {tuned['peak_allocated_mb']:.2f} MiB")
    print(f"reserved 变化：{base['peak_reserved_mb']:.2f} → {tuned['peak_reserved_mb']:.2f} MiB")
    print(f"profiler 内 step time：{base['profile_step_time_ms']:.2f} → {tuned['profile_step_time_ms']:.2f} ms（仅归因）")
    print('请在 trace 中检查 phase.forward / phase.backward / phase.optimizer 和 checkpoint 重算事件。')

if phase_report_path.is_file():
    phase_report = json.loads(phase_report_path.read_text(encoding='utf-8'))
    print('\n阶段显存归因（不用于采用门控）')
    print(json.dumps({
        'config': phase_report.get('config'),
        'summary': phase_report.get('summary'),
        'candidates': [
            {'name': item.get('name'), 'status': item.get('status'), 'global_peak_phase': item.get('global_peak_phase'), 'phase_metrics': item.get('phase_metrics')}
            for item in phase_report.get('candidates', [])
        ],
    }, ensure_ascii=False, indent=2))
elif RUN_STAGE in {'phase', 'close'}:
    print(f'尚未生成阶段显存报告：{phase_report_path}')

if close_report_path.is_file():
    close_report = json.loads(close_report_path.read_text(encoding='utf-8'))
    print('\n项目收口（采用门控来自上游无 profiler 重复测量）')
    print(json.dumps({
        'top_level_decision': close_report.get('decision', {}),
        'decision_scopes': close_report.get('decision_scopes', {}),
    }, ensure_ascii=False, indent=2))

```

#### 5.6 形成决策与交接结果

`close` 在 5.5 核对通过后生成专项报告和交接文件。相同标签和相同证据可重复运行；同一路径已有不同内容时停止。

| 输出字段 | 适用范围 |
| --- | --- |
| `decision`、`full_training_adoption` | AdamW、`seq_len=768` 的完整训练配置 |
| `activation_memory_mechanism` | 无动量 SGD 的长度阶梯 |


```python
# 只根据 5.5 已读取并校验的证据形成决策；本单元不加载模型。
if RUN_STAGE == 'close':
    if not evidence_review['ready_for_decision']:
        raise RuntimeError('证据尚未通过 5.5 核对，不能生成决策与交接资产。')
    if RUN_LABEL == 'replace_with_unique_label':
        raise ValueError('请先把 RUN_LABEL 替换为本次运行的唯一标签。')

    project_report, qualification, source_artifacts, workload_identity, decision_scopes = build_upstream_report(upstream_reports)
    if OUTPUT_PATH.exists():
        existing_report = load_json_object(OUTPUT_PATH)
        if existing_report != project_report:
            raise RuntimeError('检测到同名但内容不一致的专项报告；请审核后更换 RUN_LABEL，禁止覆盖。')
        report_status = 'already_complete'
    else:
        if HANDOFF_PATH.exists():
            raise RuntimeError('检测到 handoff 存在但专项报告缺失，资产链不完整；请先审核现有文件。')
        save_json_record_once(OUTPUT_PATH, project_report)
        report_status = 'written'

    # 热重载避免 Notebook 内核继续持有升级前、不支持 decision_scopes 的旧函数。
    performance_handoff_runtime_module = importlib.reload(performance_handoff_runtime_module)
    make_performance_handoff = performance_handoff_runtime_module.build_performance_handoff
    performance_handoff = make_performance_handoff(
        run_label=RUN_LABEL,
        workload=workload_identity,
        baseline_path=source_artifacts['baseline']['path'],
        profiling_report_path=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        qualification=qualification,
        source_artifacts={
            **source_artifacts,
            'profile_trace': relative_artifact(PROFILE_REPORT_PATH),
            'closure': relative_artifact(OUTPUT_PATH),
        },
        decision_scopes=decision_scopes,
    )
    if HANDOFF_PATH.exists():
        existing_handoff = load_json_object(HANDOFF_PATH)
        if existing_handoff != performance_handoff:
            raise RuntimeError('检测到同名但内容不一致的 handoff；请审核后更换 RUN_LABEL，禁止覆盖。')
        handoff_status = 'already_complete'
    else:
        save_json_record_once(HANDOFF_PATH, performance_handoff)
        handoff_status = 'written'

    print(json.dumps({
        'status': 'already_complete' if report_status == handoff_status == 'already_complete' else 'completed',
        'report_status': report_status,
        'handoff_status': handoff_status,
        'decision': project_report['decision'],
        'decision_scopes': project_report['decision_scopes'],
        'result_path': str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        'handoff_path': str(HANDOFF_PATH.relative_to(PROJECT_ROOT)),
    }, ensure_ascii=False, indent=2))
else:
    print(f'跳过决策与交接：当前 RUN_STAGE={RUN_STAGE!r}；只有 close 会写入最终资产。')

```

---
## 相关阅读

- [PyTorch Profiler 官方文档](https://pytorch.org/docs/stable/profiler.html)
- [Perfetto 官方文档](https://perfetto.dev/docs/)
- [Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
