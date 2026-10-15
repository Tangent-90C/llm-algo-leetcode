# 75. Memory Budget Compression Project | 显存预算压缩项目

**难度：** Hard | **环境：** CPU-only；读取 GPU 项目报告 | **标签：** `显存优化`, `预算规划`, `压缩策略` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

显存预算不是单纯追求峰值越低，而是在固定模型、数据和 workload 下，判断 checkpoint、offload 等方案能否在显存、吞吐和质量之间达到目标。本节带你从训练报告中提取可比较字段，检查证据是否适用，分析预算阈值变化对候选方案的影响，最后形成 `accept / tune / reject` 决策。

**关键词：** `memory`, `budget`, `checkpoint`, `offload`, `project`

---
## 前置阅读

**导语：** 先了解激活检查点和卸载如何改变训练状态，再查看训练测量与 checkpoint / offload 对照结果，最后进入本节的显存预算筛选与决策。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
---

### Step 1：把策略证据放入统一预算约束
本项目接收已经完成真实测量的训练策略报告，在同一训练任务下施加显存、吞吐和质量约束。它不重新运行候选，也不默认叠加所有节省手段；输出是可行候选集合、淘汰原因和优先进入回归验收的方案。

| 输入资产 | 决策操作 | 证据与交付物 |
|:---|:---|:---|
| 配对策略报告 | 核对模型、数据、精度、负载和硬件身份 | 可比较的候选集合 |
| 资源与质量约束 | 依次检查峰值显存、吞吐保留率和质量上限 | 每个候选的通过项与淘汰原因 |
| 候选代价 | 对照重算、搬运、容量节省和失败记录 | 不同资源之间的代价转移 |
| 预算决策 | 在可行集合中选择满足目标且代价最低的候选 | 最佳候选与回归验收入口 |

![75 显存预算决策流程](../public/02_PyTorch_Algorithms/75_budget_decision_flow.svg)

### Step 2：判断证据是否适用
先判断证据能否用于当前预算。实测证据只支持报告对应的硬件与训练负载；账本投影可以帮助规划下一轮批次、序列长度或候选策略，但不能证明真实吞吐、执行路径或 OOM 边界。配置发生变化时，旧报告只能作为规划参考。

| 证据类型 | 现在能回答什么 | 不能支持的结论 | 下一动作 |
|:---|:---|:---|:---|
| 同硬件同负载实测 | 当前条件下的预算可行性和候选排序 | 其他硬件或负载下的收益 | 进入阈值敏感性检查 |
| 对象账本投影 | 容量方向和下一轮实验规模 | 实际吞吐、执行路径和 OOM 边界 | 返回候选实验补测 |
| 跨硬件或跨负载记录 | 可能的策略方向 | 当前预算下的采用结论 | 重新建立同口径基线与候选 |



### Step 3：用阈值敏感性检查预算结论
固定同一份有效候选报告，只改变一项预算阈值，观察可行集合、最佳候选和淘汰原因是否变化。若轻微调整阈值就改变最佳候选，结论应标为敏感并进入补测；若候选在一段合理阈值范围内保持可行，才适合进入回归验收。



| 检查项 | 调整方式 | 观察结果 |
|:---|:---|:---|
| 显存上限 | 逐步收紧可用容量 | 可行候选和最佳候选是否改变 |
| 吞吐下限 | 逐步提高吞吐保留率 | 哪些策略因速度门槛被淘汰 |
| 质量上限 | 逐步收紧验证损失上限 | 质量约束是否改变候选排序 |
| 结论稳定性 | 综合比较不同阈值下的结果 | 判断预算结论是否稳定 |

![75 预算敏感性流程](../public/02_PyTorch_Algorithms/75_sensitivity_flow.svg)

### Step 4：实现预算筛选与敏感性决策
请补全 3 个函数：先确认 76 移交的证据适用于当前任务，再把候选放入统一预算，最后检查更严格阈值下的选择是否稳定。字段读取和预算数值检查已提供，题目聚焦证据适用性与决策。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：核验证据适用性 | 76 的移交记录与当前任务身份 | 仅接收工作负载、硬件和状态一致的配对证据 | 错误工作负载与未就绪状态被拒绝 |
| TODO 2：执行预算筛选 | 可比较候选与显存、吞吐、质量门槛 | 每个候选保留通过状态或明确淘汰原因 | 显存、吞吐和质量门槛分别被覆盖 |
| TODO 3：检查选择稳定性 | 基准预算与更严格预算下的筛选结果 | 区分稳定采用、阈值敏感和无可行候选 | 候选变化时不能直接采用 |



```python
import math
from typing import Dict, List

```


```python
# 3 个核心 TODO：证据适用性、预算筛选、敏感性决策。
# 预算数值检查由辅助函数完成；学习者负责判断证据能否用于当前决策。

def _budget_is_valid(budget: Dict[str, float]) -> bool:
    required = ('memory_cap_mb', 'min_memory_saving_mb', 'min_throughput_ratio', 'max_quality_delta')
    return all(
        isinstance(budget.get(key), (int, float)) and math.isfinite(budget[key])
        for key in required
    ) and budget['memory_cap_mb'] > 0 and budget['min_memory_saving_mb'] >= 0 and 0 <= budget['min_throughput_ratio'] <= 1 and budget['max_quality_delta'] >= 0

def validate_budget_evidence(handoff: Dict[str, object], workload_id: str, hardware_id: str) -> Dict[str, object]:
    """确认上游证据可以用于当前预算决策。"""
    # ==========================================
    # TODO 1：核验证据适用性。
    # 检查移交状态、工作负载、硬件和候选集合；返回 is_applicable 与 issues。
    # issues = ???
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def evaluate_budget_candidates(candidates: List[Dict[str, object]], budget: Dict[str, float]) -> Dict[str, object]:
    """在统一预算下筛选候选，并保留每个淘汰原因。"""
    # ==========================================
    # TODO 2：执行预算筛选。
    # 对 comparable 候选检查绝对显存、最低节省、吞吐保留率和质量变化。
    # evaluations = ???
    # feasible = ???
    # 返回 best_candidate、feasible_names 与 evaluations。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_budget_sensitivity(base_result: Dict[str, object], stricter_results: List[Dict[str, object]]) -> Dict[str, object]:
    """根据基准预算和更严格预算下的候选变化形成项目决策。"""
    # ==========================================
    # TODO 3：检查选择稳定性。
    # 基准预算无候选时 reject；更严格预算改变赢家或淘汰赢家时 tune；赢家稳定时 accept。
    # selected_candidates = ???
    # 返回 decision、reason、selected_candidate 与 next_action。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_memory_budget_project():
    try:
        candidates = [
            {'name': 'checkpoint', 'status': 'comparable', 'candidate_peak_memory_mb': 8200.0, 'memory_saving_mb': 2800.0, 'throughput_ratio': 0.85, 'quality_delta': 0.02},
            {'name': 'offload', 'status': 'comparable', 'candidate_peak_memory_mb': 7600.0, 'memory_saving_mb': 3400.0, 'throughput_ratio': 0.62, 'quality_delta': 0.01},
        ]
        handoff = {
            'status': 'ready_for_budget_review', 'workload_id': 'train-b2-s1024',
            'hardware_id': 'gpu-a', 'eligible_candidates': ['checkpoint', 'offload'],
            'candidates': candidates,
        }
        check = validate_budget_evidence(handoff, 'train-b2-s1024', 'gpu-a')
        assert check['is_applicable'] is True
        mismatch = validate_budget_evidence(handoff, 'train-b4-s1024', 'gpu-a')
        assert mismatch['is_applicable'] is False and 'workload_mismatch' in mismatch['issues']

        budget = {'memory_cap_mb': 9000.0, 'min_memory_saving_mb': 2000.0, 'min_throughput_ratio': 0.75, 'max_quality_delta': 0.03}
        base_result = evaluate_budget_candidates(candidates, budget)
        assert base_result['best_candidate'] == 'checkpoint'
        evaluations = {item['name']: item for item in base_result['evaluations']}
        assert evaluations['offload']['reasons'] == ['throughput_below_floor']
        stricter_same = evaluate_budget_candidates(candidates, {**budget, 'memory_cap_mb': 8500.0})
        stable = decide_budget_sensitivity(base_result, [stricter_same])
        assert stable['decision'] == 'accept' and stable['selected_candidate'] == 'checkpoint'

        stricter_changed = evaluate_budget_candidates(candidates, {**budget, 'min_throughput_ratio': 0.90})
        sensitive = decide_budget_sensitivity(base_result, [stricter_changed])
        assert sensitive['decision'] == 'tune' and sensitive['reason'] == 'selection_is_threshold_sensitive'
        empty = evaluate_budget_candidates(candidates, {**budget, 'memory_cap_mb': 7000.0})
        assert decide_budget_sensitivity(empty, [])['decision'] == 'reject'
        print('所有测试通过！')
    except NotImplementedError:
        print('请先完成 TODO 代码！')
        raise
    except AssertionError as e:
        print(f'测试失败: {e}')
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except Exception as e:
        print(f'发生非预期错误: {e}')
        raise


test_memory_budget_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 完整参考实现
答案区保留一份可独立运行的内联实现，用于对照题目区 TODO；Step 5 的项目执行代码会调用同逻辑的公共 runtime。


```python
# 完整参考实现：与题目区的 3 个 TODO 一一对应；不替代 Step 5 的公共 runtime。
import math
from typing import Dict, List

def validate_memory_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """检查预算字段和值，返回校验状态、缺失字段和非法字段。

    不为缺失字段补默认值；显存、吞吐、收益和质量阈值的单位由输入报告约定。
    """
    # ==========================================
    # TODO 1 对应变量 required_budget_keys：收集预算必需字段。
    # TODO 1 对应变量 missing_keys / invalid_keys：分别记录缺失字段和非法字段。
    # 提示：required_budget_keys / required_quality_keys 定义输入契约；
    # missing_keys 记录缺失字段，numeric_values 汇总待检查值，invalid_keys 记录非法字段。
    # ==========================================
    required_budget_keys = ['memory_cap_mb', 'min_samples_per_s', 'min_memory_saving_mb', 'min_throughput_ratio']  # 预算必需字段
    required_quality_keys = ['max_val_loss']
    missing_keys = [key for key in required_budget_keys if key not in budget]  # 缺失字段不参与数值校验
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    numeric_values = {key: budget.get(key) for key in required_budget_keys}
    numeric_values.update({key: quality_floor.get(key) for key in required_quality_keys})
    # numeric_values 统一收集待检查值，invalid_keys 只记录字段名，便于测试和报告读取。
    invalid_keys = [
        key for key, value in numeric_values.items()
        if key not in missing_keys and (not isinstance(value, (int, float)) or not math.isfinite(value))
    ]
    memory_cap = budget.get('memory_cap_mb')
    min_throughput = budget.get('min_samples_per_s')
    min_saving = budget.get('min_memory_saving_mb')
    throughput_ratio = budget.get('min_throughput_ratio')
    max_loss = quality_floor.get('max_val_loss')
    if isinstance(memory_cap, (int, float)) and memory_cap <= 0:
        invalid_keys.append('memory_cap_mb')
    if isinstance(min_throughput, (int, float)) and min_throughput < 0:
        invalid_keys.append('min_samples_per_s')
    if isinstance(min_saving, (int, float)) and min_saving < 0:
        invalid_keys.append('min_memory_saving_mb')
    if isinstance(throughput_ratio, (int, float)) and not 0 <= throughput_ratio <= 1:
        invalid_keys.append('min_throughput_ratio')
    if isinstance(max_loss, (int, float)) and max_loss < 0:
        invalid_keys.append('max_val_loss')
    invalid_keys = list(dict.fromkeys(invalid_keys))
    return {
        'is_valid': len(missing_keys) == 0 and len(invalid_keys) == 0,
        'missing_keys': missing_keys,
        'invalid_keys': invalid_keys,
    }


def summarize_memory_strategies(candidates: List[Dict[str, object]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """筛选候选策略，保留淘汰原因，并计算相对 baseline 的收益。

    OOM、重复名称和缺失指标分别记录，不把失败实验伪装成零值。
    """
    # ==========================================
    # TODO 2 对应变量 evaluations / feasible：记录候选状态并保留可行候选。
    # TODO 2 对应派生判断 memory_ok / speed_ok / quality_ok：对应显存、吞吐和质量门槛。
    # 提示：先处理 OOM、重复名称和无效指标，再用 memory_ok / speed_ok / quality_ok 筛选。
    # baseline 只用于计算显存节省和吞吐保留率，不参与候选排序。
    # ==========================================
    feasible: List[Dict[str, object]] = []  # 通过全部门槛的候选
    quality_failed = 0
    invalid_count = 0
    oom_count = 0
    evaluations = []
    seen_names = set()

    for candidate in candidates:
        # 先处理 OOM、重复名称和无效指标，再检查三个预算门槛。
        name = candidate.get('name')
        if not isinstance(name, str) or not name or name in seen_names:
            invalid_count += 1
            evaluations.append({'name': name, 'status': 'invalid', 'reasons': ['missing_or_duplicate_name']})
            continue
        seen_names.add(name)
        if candidate.get('status', 'ok') == 'oom':
            oom_count += 1
            evaluations.append({'name': candidate.get('name'), 'status': 'oom', 'reasons': ['oom']})
            continue
        memory = candidate.get('peak_memory_mb')
        throughput = candidate.get('samples_per_s')
        eval_loss = candidate.get('eval_loss', candidate.get('val_loss'))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            evaluations.append({'name': candidate.get('name'), 'status': 'invalid', 'reasons': ['missing_or_non_numeric_metric']})
            continue
        # 三个布尔变量分别对应显存、吞吐和质量门槛，便于生成可解释的 reasons。
        memory_ok = memory <= budget['memory_cap_mb']
        speed_ok = throughput >= budget['min_samples_per_s']
        quality_ok = eval_loss <= quality_floor['max_val_loss']
        if not quality_ok:
            quality_failed += 1
        reasons = []
        if not memory_ok:
            reasons.append('memory_over_budget')
        if not speed_ok:
            reasons.append('throughput_below_floor')
        if not quality_ok:
            reasons.append('quality_over_floor')
        evaluations.append({'name': candidate.get('name'), 'status': 'feasible' if not reasons else 'rejected', 'reasons': reasons})
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    # baseline 只用于计算节省和吞吐保留率；候选排序不依赖 baseline。
    feasible.sort(key=lambda x: (x['peak_memory_mb'], -x['samples_per_s'], x.get('eval_loss', x.get('val_loss'))))
    best_candidate = feasible[0]['name'] if feasible else None
    baseline = next((item for item in candidates if item.get('name') == 'baseline' and item.get('status', 'ok') == 'ok' and all(isinstance(item.get(key), (int, float)) and math.isfinite(item.get(key)) for key in ('peak_memory_mb', 'samples_per_s'))), None)
    best = feasible[0] if feasible else None
    return {
        'candidate_count': len(candidates),
        'measured_count': len(candidates) - oom_count - invalid_count,
        'oom_count': oom_count,
        'invalid_count': invalid_count,
        'feasible_count': len(feasible),
        'best_candidate': best_candidate,
        'quality_failed_count': quality_failed,
        'feasible_names': [item['name'] for item in feasible],
        'baseline_peak_memory_mb': baseline['peak_memory_mb'] if baseline else None,
        'throughput_ratio': (best['samples_per_s'] / baseline['samples_per_s']) if baseline and best else None,
        'baseline_available': baseline is not None,
        'best_peak_memory_mb': best['peak_memory_mb'] if best else None,
        'memory_saving_mb': (baseline['peak_memory_mb'] - best['peak_memory_mb']) if baseline and best else 0.0,
        'min_memory_saving_mb': budget['min_memory_saving_mb'],
        'min_throughput_ratio': budget['min_throughput_ratio'],
        'evaluations': evaluations,
    }


def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据候选摘要和收益阈值输出 accept、tune 或 reject。

    返回 decision、reason 和 next_action，供项目报告记录下一步动作。
    """
    # ==========================================
    # TODO 3 对应变量 feasible_count：确认是否存在可行候选。
    # TODO 3 对应派生判断 meaningful_memory_gain / acceptable_throughput：检查收益门槛。
    # TODO 3 对应结果 decision / reason / next_action：收束为项目结论。
    # 提示：先读取 baseline_available / feasible_count，再判断收益布尔量，最后返回三个结论字段。
    # ==========================================
    feasible_count = summary['feasible_count']  # 可行候选数量
    best_candidate = summary['best_candidate']
    quality_failed_count = summary['quality_failed_count']

    if not summary.get('baseline_available', False):
        return {
            'decision': 'reject',
            'reason': 'baseline_missing_or_invalid',
            'next_action': 'rerun_baseline_before_comparing_candidates',
        }
    if feasible_count == 0:
        return {
            'decision': 'reject',
            'reason': 'no_strategy_meets_budget_and_quality',
            'next_action': 'tighten_batch_or_rework_memory_plan',
        }
    # 两个布尔量分别判断显存收益和吞吐保留率是否达到预算门槛。
    meaningful_memory_gain = summary.get('memory_saving_mb', 0.0) >= summary['min_memory_saving_mb']
    acceptable_throughput = summary.get('throughput_ratio') is not None and summary['throughput_ratio'] >= summary['min_throughput_ratio']
    if best_candidate != 'baseline' and meaningful_memory_gain and acceptable_throughput:
        return {
            'decision': 'accept',
            'reason': 'compression_strategy_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'compression_needs_quality_recovery',
            'next_action': 'adjust_checkpoint_scope_or_batch_plan',
        }
    if not meaningful_memory_gain:
        return {
            'decision': 'tune',
            'reason': 'memory_saving_below_meaningful_threshold',
            'next_action': 'test_larger_pressure_or_optimizer_compression',
        }
    if not acceptable_throughput:
        return {
            'decision': 'tune',
            'reason': 'throughput_loss_exceeds_budget',
            'next_action': 'reduce_compression_scope',
        }
    return {
        'decision': 'tune',
        'reason': 'baseline_still_best_under_current_budget',
        'next_action': 'revisit_memory_strategy_mix',
    }


def _budget_is_valid(budget: Dict[str, float]) -> bool:
    required = ('memory_cap_mb', 'min_memory_saving_mb', 'min_throughput_ratio', 'max_quality_delta')
    return all(isinstance(budget.get(key), (int, float)) and math.isfinite(budget[key]) for key in required) and budget['memory_cap_mb'] > 0 and budget['min_memory_saving_mb'] >= 0 and 0 <= budget['min_throughput_ratio'] <= 1 and budget['max_quality_delta'] >= 0

def validate_budget_evidence(handoff: Dict[str, object], workload_id: str, hardware_id: str) -> Dict[str, object]:
    issues = []
    if handoff.get('status') != 'ready_for_budget_review':
        issues.append('handoff_not_ready')
    if handoff.get('workload_id') != workload_id:
        issues.append('workload_mismatch')
    if handoff.get('hardware_id') != hardware_id:
        issues.append('hardware_mismatch')
    eligible = handoff.get('eligible_candidates')
    if not isinstance(eligible, list) or not eligible:
        issues.append('eligible_candidates_missing')
    return {'is_applicable': not issues, 'issues': issues}

def evaluate_budget_candidates(candidates: List[Dict[str, object]], budget: Dict[str, float]) -> Dict[str, object]:
    if not _budget_is_valid(budget):
        raise ValueError('预算字段缺失或取值非法')
    evaluations = []
    feasible = []
    for candidate in candidates:
        name = candidate.get('name')
        if candidate.get('status') != 'comparable':
            evaluations.append({'name': name, 'status': 'excluded', 'reasons': ['evidence_not_comparable']})
            continue
        reasons = []
        if candidate.get('candidate_peak_memory_mb', math.inf) > budget['memory_cap_mb']:
            reasons.append('memory_over_budget')
        if candidate.get('memory_saving_mb', -math.inf) < budget['min_memory_saving_mb']:
            reasons.append('memory_saving_below_floor')
        if candidate.get('throughput_ratio', -math.inf) < budget['min_throughput_ratio']:
            reasons.append('throughput_below_floor')
        if candidate.get('quality_delta', math.inf) > budget['max_quality_delta']:
            reasons.append('quality_delta_over_limit')
        evaluations.append({'name': name, 'status': 'feasible' if not reasons else 'rejected', 'reasons': reasons})
        if not reasons:
            feasible.append(candidate)
    feasible.sort(key=lambda item: (item['candidate_peak_memory_mb'], -item['throughput_ratio'], item['quality_delta']))
    return {
        'best_candidate': feasible[0]['name'] if feasible else None,
        'feasible_names': [item['name'] for item in feasible],
        'evaluations': evaluations,
    }

def decide_budget_sensitivity(base_result: Dict[str, object], stricter_results: List[Dict[str, object]]) -> Dict[str, object]:
    selected = base_result.get('best_candidate')
    if selected is None:
        return {'decision': 'reject', 'reason': 'no_candidate_meets_base_budget', 'selected_candidate': None, 'next_action': 'rework_strategy_or_budget'}
    selected_candidates = [result.get('best_candidate') for result in stricter_results]
    if any(candidate != selected for candidate in selected_candidates):
        return {'decision': 'tune', 'reason': 'selection_is_threshold_sensitive', 'selected_candidate': selected, 'next_action': 'collect_boundary_evidence'}
    return {'decision': 'accept', 'reason': 'selection_stable_across_thresholds', 'selected_candidate': selected, 'next_action': 'send_to_regression_acceptance'}

```

### 解析

**1. 机制：核验证据适用性**
- 预算判断只能消费已经完成配对测量、且与当前工作负载和硬件一致的候选证据。
- 不变量：未就绪或跨工作负载的移交记录不得进入预算筛选。
- 测试证据：测试覆盖正确移交和工作负载不一致两种情况。

**2. 机制：执行统一预算筛选**
- 每个候选同时接受绝对显存、最低节省、吞吐保留率和质量变化检查，淘汰原因保留在证据中。
- 不变量：显存更低不能抵消吞吐或质量越界。
- 测试证据：测试确认 offload 因吞吐不足被淘汰，而 checkpoint 保持可行。

**3. 机制：检查阈值敏感性**
- 基准预算下没有候选时拒绝；更严格阈值改变赢家时继续调优；赢家保持一致时才移交给 74 做回归验收。
- 不变量：只在单一阈值组合下胜出的候选不能直接采用。
- 测试证据：测试覆盖稳定采用、阈值敏感和无候选三种结果。

### Step 5（项目实验）：生成显存预算决策

本实验读取 76 保存的重复测量结果，在不重新训练的情况下检查显存、吞吐、质量和显存收益门槛。运行顺序是 `prepare → analyze → close`：先核对输入，再生成预算证据，最后写入供 74 使用的决策结果。

| 阶段 | 操作 | 结果 |
| --- | --- | --- |
| `prepare` | 核对固定输入和已登记结果 | 输入摘要 |
| `analyze` | 执行预算网格和敏感性分析 | 原始预算证据 |
| `close` | 核对原始证据并形成结论 | 决策 JSON |

#### 5.1 本次运行配置

只需填写本次标签和运行阶段。模型、上游结果、预算和敏感性网格在 5.2 固定。


```python
# 只修改本单元。新实验使用唯一标签，并按 prepare → analyze → close 依次运行。
RUN_LABEL = 'replace_with_unique_label'
RUN_STAGE = 'prepare'  # prepare / analyze / close

```

#### 5.2 固定实验配置与输入预检

本单元定位项目根目录，读取固定证据清单，并核对 76 的结果文件。它不启动模型，也不写结果。


```python
from pathlib import Path
import hashlib
import json
import os
import sys

PROJECT_ROOT = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
PROJECT_ROOT = next(
    (candidate for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents) if (candidate / 'tools/project_runtime.py').is_file()),
    PROJECT_ROOT,
)
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'tools/project_runtime.py').is_file():
        PROJECT_ROOT = colab_root
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    raise RuntimeError('找不到项目根目录：请设置 LLM_ALGO_PROJECT_ROOT，或先把仓库放到 /content/llm-algo-leetcode。')
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if RUN_STAGE not in {'prepare', 'analyze', 'close'}:
    raise ValueError(f'未知 RUN_STAGE: {RUN_STAGE}')
if RUN_STAGE != 'prepare' and RUN_LABEL == 'replace_with_unique_label':
    raise ValueError('analyze / close 需要唯一 RUN_LABEL。')

SEMANTIC_ID = 'PERF-MEMORY-BUDGET'
SCHEMA_VERSION = 'training-performance-project/v1'
INPUT_MANIFEST_RELATIVE_PATH = Path('benchmarks/results/75_input_manifest_rtx5070ti_memory_budget_v1.json')
INPUT_MANIFEST_PATH = PROJECT_ROOT / INPUT_MANIFEST_RELATIVE_PATH
input_manifest = json.loads(INPUT_MANIFEST_PATH.read_text(encoding='utf-8'))
source_spec = input_manifest['source']
approved_spec = input_manifest['approved_result']
RESULT_76_RELATIVE_PATH = Path(source_spec['path'])
REPORT_PATH = PROJECT_ROOT / RESULT_76_RELATIVE_PATH

HARDWARE_PROFILE = {
    'name': '12GB Laptop GPU',
    'total_memory_mb': 12227,
    'usable_memory_mb': 11200,
}
BUDGET = {
    'memory_cap_mb': float(HARDWARE_PROFILE['usable_memory_mb']),
    'min_samples_per_s': 1.0,
    'min_memory_saving_mb': 512.0,
    'min_throughput_ratio': 0.70,
}
MAX_VAL_LOSS = None
SENSITIVITY_MEMORY_CAPS = [9600.0, 11200.0]
SENSITIVITY_THROUGHPUT_FLOORS = [1.0, 1.5, 2.0]
SENSITIVITY_QUALITY_FLOORS = [12.20, 12.45, 12.80]
MODEL_PROFILE = {
    'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
    'task': 'causal_lm_training',
    'dataset_name': '固定小型教学数据集',
    'dataset_samples': 4,
}
EVIDENCE_OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/75_memory_budget_evidence_{RUN_LABEL}.json')
OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/75_memory_budget_decision_{RUN_LABEL}.json')

source_sha256 = hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest() if REPORT_PATH.is_file() else None
source_payload = json.loads(REPORT_PATH.read_text(encoding='utf-8')) if REPORT_PATH.is_file() else {}
output_parent = PROJECT_ROOT / OUTPUT_RELATIVE_PATH.parent
preflight_report = {
    'project_root': str(PROJECT_ROOT),
    'run_stage': RUN_STAGE,
    'source_path': str(RESULT_76_RELATIVE_PATH),
    'source_exists': REPORT_PATH.is_file(),
    'source_sha256': source_sha256,
    'source_hash_matches': source_sha256 == source_spec['sha256'],
    'source_schema': source_payload.get('schema_version'),
    'source_semantic_id': source_payload.get('semantic_id'),
    'output_parent_exists': output_parent.is_dir(),
    'output_parent_writable': output_parent.is_dir() and os.access(output_parent, os.W_OK),
}
preflight_report['ready'] = (
    preflight_report['source_hash_matches']
    and preflight_report['source_schema'] == source_spec['schema_version']
    and preflight_report['source_semantic_id'] == source_spec['semantic_id']
    and preflight_report['output_parent_writable']
)
print(preflight_report)
if not preflight_report['ready']:
    raise RuntimeError('输入预检未通过，未执行预算分析。')

```

#### 5.3 输入合同与输出路径

核对 76 的模型、工作负载、候选和重复次数，并派生本次结果路径。


```python
source_config = source_payload.get('config', {})
required_source_fields = ('model_id', 'workload', 'batch_size', 'seq_len', 'dtype', 'optimizer')
missing_source_fields = [key for key in required_source_fields if source_config.get(key) in (None, '')]
candidate_names = [item.get('name') for item in source_payload.get('candidates', [])]
missing_repeats = [
    item.get('name', '<unknown>')
    for item in source_payload.get('candidates', [])
    if len(item.get('runs', [])) < 3
]
if missing_source_fields or not candidate_names or 'baseline' not in candidate_names or missing_repeats:
    raise ValueError({
        'missing_source_fields': missing_source_fields,
        'candidate_names': candidate_names,
        'missing_three_repeats': missing_repeats,
    })

DECISION_IDENTITY = {
    'run_label': RUN_LABEL,
    'run_stage': RUN_STAGE,
    'budget': BUDGET,
    'source': str(RESULT_76_RELATIVE_PATH),
    'sensitivity_scenarios': len(SENSITIVITY_MEMORY_CAPS) * len(SENSITIVITY_THROUGHPUT_FLOORS) * len(SENSITIVITY_QUALITY_FLOORS),
    'evidence_path': str(EVIDENCE_OUTPUT_RELATIVE_PATH),
    'decision_path': str(OUTPUT_RELATIVE_PATH),
}
print(DECISION_IDENTITY)

```

#### 5.4 生成预算证据

`analyze` 读取 76 的配对结果，执行预算筛选和敏感性分析。`prepare` 与 `close` 不在本单元写文件。


```python
import json
from tools.project_runtime import ensure_output_path, require_input_file, save_json_record_once, standard_experiment_config, standard_training_metrics
from tools.memory_budget_runtime import summarize_memory_strategies, decide_memory_budget_project

RESULT_76_PATH = PROJECT_ROOT / RESULT_76_RELATIVE_PATH
if RUN_STAGE == 'analyze':
    require_input_file(RESULT_76_PATH, '76 结果')
    raw = json.loads(RESULT_76_PATH.read_text(encoding='utf-8'))
    source_config = raw['config']
    candidates = []
    for item in raw['candidates']:
        candidates.append({
            'name': item['name'],
            'status': item.get('status', 'ok'),
            'step_time_ms': item.get('step_time_ms'),
            'samples_per_s': item.get('samples_per_s'),
            'peak_memory_mb': item.get('peak_memory_mb', item.get('peak_mem_mb')),
            'peak_reserved_mb': item.get('peak_reserved_mb'),
            'loss': item.get('loss'),
            'eval_loss': item.get('eval_loss', item.get('val_loss')),
            'val_loss': item.get('val_loss', item.get('eval_loss')),
            'error': item.get('error'),
        })
    baseline = next((x for x in candidates if x['name'] == 'baseline'), None)
    if baseline is None:
        raise RuntimeError('76 没有可用 baseline，无法形成预算决策。')
    max_val_loss = baseline['eval_loss'] * 1.02 if MAX_VAL_LOSS is None else MAX_VAL_LOSS
    quality_floor = {'max_val_loss': round(max_val_loss, 6)}
    summary = summarize_memory_strategies(candidates, BUDGET, quality_floor)
    sensitivity = []
    for memory_cap_mb in SENSITIVITY_MEMORY_CAPS:
        for throughput_floor in SENSITIVITY_THROUGHPUT_FLOORS:
            for quality_floor_value in SENSITIVITY_QUALITY_FLOORS:
                sensitivity_budget = {
                    'memory_cap_mb': memory_cap_mb,
                    'min_samples_per_s': throughput_floor,
                    'min_memory_saving_mb': BUDGET['min_memory_saving_mb'],
                    'min_throughput_ratio': BUDGET['min_throughput_ratio'],
                }
                sensitivity_quality_floor = {'max_val_loss': quality_floor_value}
                sensitivity_summary = summarize_memory_strategies(candidates, sensitivity_budget, sensitivity_quality_floor)
                sensitivity_decision = decide_memory_budget_project(sensitivity_summary)
                sensitivity.append({
                    'memory_cap_mb': memory_cap_mb,
                    'min_samples_per_s': throughput_floor,
                    'max_val_loss': quality_floor_value,
                    'feasible_names': sensitivity_summary['feasible_names'],
                    'best_candidate': sensitivity_summary['best_candidate'],
                    'decision': sensitivity_decision['decision'],
                })
    decision_counts = {}
    best_candidate_counts = {}
    feasible_sets = {}
    for row in sensitivity:
        scenario_decision = row['decision']
        decision_counts[scenario_decision] = decision_counts.get(scenario_decision, 0) + 1
        best_name = row['best_candidate'] or '<none>'
        best_candidate_counts[best_name] = best_candidate_counts.get(best_name, 0) + 1
        feasible_key = ','.join(row['feasible_names']) or '<none>'
        feasible_sets[feasible_key] = feasible_sets.get(feasible_key, 0) + 1
    sensitivity_summary = {
        'scenario_count': len(sensitivity),
        'decision_counts': decision_counts,
        'best_candidate_counts': best_candidate_counts,
        'feasible_set_counts': feasible_sets,
        'best_candidate_stable': len(best_candidate_counts) == 1,
        'feasible_set_stable': len(feasible_sets) == 1,
        'decision_stable': len(decision_counts) == 1,
        'interpretation': 'stable' if len(best_candidate_counts) == 1 and len(feasible_sets) == 1 else 'sensitive_to_thresholds',
    }
    budget_evidence = {
        'schema_version': 'memory-budget-evidence/v1', 'semantic_id': SEMANTIC_ID,
        'run_mode': 'measured_read_only', 'run_label': RUN_LABEL,
        'task': 'task3_training_memory_optimization',
        'stage': 'memory_budget_evidence',
        'source': str(RESULT_76_PATH.relative_to(PROJECT_ROOT)),
        'budget': BUDGET,
        'quality_floor': quality_floor,
        'evidence_mode': 'measured',
        'planning_context': {
            'model': MODEL_PROFILE,
            'hardware_target': HARDWARE_PROFILE,
            'note': '硬件目标用于预算规划；只有 76 报告中的硬件和 workload 属于 measured evidence。',
        },
        'candidates': candidates,
        'summary': summary,
        'sensitivity': sensitivity,
        'sensitivity_summary': sensitivity_summary,
    }
    budget_evidence['planning_context']['measured_hardware'] = source_config.get('device')
    budget_evidence['planning_context']['hardware_comparison'] = 'same_hardware_required_for_performance_claim'
    budget_evidence['experiment'] = standard_experiment_config({**source_config, 'dtype': source_config.get('dtype', source_config.get('amp_dtype'))})
    budget_evidence['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    evidence_output_path = ensure_output_path(PROJECT_ROOT, EVIDENCE_OUTPUT_RELATIVE_PATH)
    save_json_record_once(evidence_output_path, budget_evidence)
    print({'saved': str(evidence_output_path), 'candidate_count': len(candidates), 'summary': summary, 'sensitivity_summary': sensitivity_summary})
else:
    print(f'跳过 5.4：RUN_STAGE={RUN_STAGE}')

```

#### 5.5 核对预算证据

`prepare` 读取已登记决策；`analyze` 和 `close` 读取当前标签的原始证据。本单元只读文件。


```python
if RUN_STAGE == 'prepare':
    review_path = PROJECT_ROOT / approved_spec['path']
    expected_review_sha256 = approved_spec['sha256']
else:
    review_path = PROJECT_ROOT / EVIDENCE_OUTPUT_RELATIVE_PATH
    expected_review_sha256 = None
if not review_path.is_file():
    raise FileNotFoundError(f'找不到预算证据：{review_path}')

budget_evidence_read = json.loads(review_path.read_text(encoding='utf-8'))
review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
review_schema = budget_evidence_read.get('schema_version')
evidence_review = {
    'path': str(review_path.relative_to(PROJECT_ROOT)),
    'sha256': review_sha256,
    'hash_matches': expected_review_sha256 is None or review_sha256 == expected_review_sha256,
    'schema_version': review_schema,
    'semantic_id': budget_evidence_read.get('semantic_id'),
    'run_label': budget_evidence_read.get('run_label'),
    'candidate_names': [item.get('name') for item in budget_evidence_read.get('candidates', [])],
    'summary': budget_evidence_read.get('summary'),
    'sensitivity_summary': budget_evidence_read.get('sensitivity_summary'),
    'final_asset': review_schema == SCHEMA_VERSION,
}
evidence_review['ready_for_decision'] = (
    evidence_review['hash_matches']
    and review_schema in {'memory-budget-evidence/v1', SCHEMA_VERSION}
    and evidence_review['semantic_id'] == SEMANTIC_ID
    and bool(evidence_review['candidate_names'])
    and isinstance(evidence_review['summary'], dict)
)
print(evidence_review)
if not evidence_review['ready_for_decision']:
    raise RuntimeError('预算证据核对未通过。')

```

#### 5.6 形成决策结果

`close` 根据 5.5 已核对的原始证据写入最终结果。相同内容可重复运行；同一路径出现不同内容时停止并要求更换标签。


```python
if RUN_STAGE == 'close':
    decision = decide_memory_budget_project(budget_evidence_read['summary'])
    project_result = {
        **budget_evidence_read,
        'schema_version': SCHEMA_VERSION,
        'stage': 'memory_budget_decision',
        'decision': decision,
        'source_evidence': {'path': evidence_review['path'], 'sha256': evidence_review['sha256']},
    }
    output_path = PROJECT_ROOT / OUTPUT_RELATIVE_PATH
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding='utf-8'))
        if existing != project_result:
            raise FileExistsError(f'结果已存在且内容不同，请更换 RUN_LABEL：{output_path}')
        print({'status': 'already_complete', 'path': str(output_path), 'decision': existing.get('decision')})
    else:
        save_json_record_once(ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH), project_result)
        print({'status': 'saved', 'path': str(output_path), 'decision': decision})
elif RUN_STAGE == 'prepare':
    print({'status': 'prepared', 'approved_result': evidence_review['path'], 'decision': budget_evidence_read.get('decision')})
else:
    print({'status': 'analyzed', 'evidence': evidence_review['path'], 'next_stage': 'close'})

```

---
## 相关阅读

- [ZeRO 论文：Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [QLoRA 论文：Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
- [DeepSpeed 官方仓库](https://github.com/microsoft/DeepSpeed)
- [PyTorch Activation Checkpointing 官方文档](https://pytorch.org/docs/stable/checkpoint.html)
- [PyTorch FSDP 官方文档](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed ZeRO-3 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [PyTorch CUDA 内存管理文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [73 训练性能分析](./73_Training_Performance_Analysis.md)
- [76 Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
