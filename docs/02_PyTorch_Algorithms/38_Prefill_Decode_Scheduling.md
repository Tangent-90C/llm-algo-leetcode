# 38. Prefill Decode Scheduling | Prefill/Decode 调度
**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `Serving`, `Chunked Prefill`, `PD Disaggregation` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

长 prompt 到达时，完整 Prefill 可能占住一整轮批次，使正在生成的请求等待更久。Chunked Prefill 先把未处理输入切成可让出的执行块；当 Prefill 与 Decode 的资源压力长期不同，再考虑把它们组织成独立服务池。

本节建立“请求压力 → 分块推进 → 分池与交接 → 服务决策”的链路。它先解释调度如何减少请求间的相互阻塞，再判断 PD 分离的吞吐和延迟收益能否覆盖交接成本。

**关键词：** `chunked prefill`, `continuous batching`, `PD disaggregation`, `handoff`
## 前置阅读

**导语：** 先能区分 Prefill 的突发输入处理与 Decode 的持续 KV 读取，再观察未命中 suffix 为什么要按可让出的粒度进入批次。

- [24. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./24_Prefix_Cache_Matching_and_Reuse.md)
- [INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界](./INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md)
- [36. Decode Scheduling | Decode 调度](./36_Decode_Scheduling.md)
- [37. KV Cache Lifecycle and Capacity Control | KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)
- [INF-CONTINUOUS-BATCHING. Continuous Batching | 连续批处理](./INF-CONTINUOUS-BATCHING_Continuous_Batching.md)
### Step 1: 请求为什么需要同时考虑 Prefill 与 Decode

一次请求先处理完整输入，再逐 token 生成输出。长 prompt 往往带来集中的 Prefill 计算和访存；长生成则长期占用 KV Cache，并对 Decode 等待敏感。Continuous Batching 已负责在每轮重组请求成员，本节进一步决定 ready 请求中的 Prefill suffix 和 Decode 工作如何共享 `max_num_batched_tokens`：本轮 Prefill chunk 的 token 数与 Decode 请求数之和不能超过这份预算。

| 请求形态 | 更突出的压力 | 首先观察什么 |
|---|---|---|
| 长输入、短生成 | Prefill 计算、输入访存与批次占用 | TTFT、Prefill 队列、Decode 抖动 |
| 短输入、长生成 | KV Cache 驻留、逐步 Decode 与排队 | TPOT、KV 使用量、Decode 队列 |
| 输入和生成都长 | 两类压力叠加 | E2E、P95/P99、池利用率 |

![Continuous Batching：Chunked Prefill 与 Decode 的调度时间线](../public/02_PyTorch_Algorithms/38_prefill_decode_timeline.svg)

### Step 2: Chunked Prefill 如何保留 Decode 的准入机会

Prefix Cache 未命中的 suffix 仍要 Prefill。Chunked Prefill 将它切成能进入本轮 token 预算的块：每完成一个 chunk，调度器便回到 ready 集合，决定继续当前请求还是先服务等待中的 Decode。分块既限制单轮工作量，也增加重新选择请求的机会；prefix 是否命中仍由缓存机制决定。

| 调度对象 | 含义 | 影响 |
|---|---|---|
| `suffix` | 本次仍需处理的未命中输入 | 决定新增 Prefill 工作量 |
| `chunk_size` | 单次进入 Prefill 的最大 token 数 | 必须服从剩余 token budget，控制单轮占用和让出频率 |
| `chunk_plan` | suffix 的有序分块计划 | 让调度器在块边界重新选请求 |
| `iteration_plan` | 本轮 Decode token 与 Prefill chunk 的组合 | 保证总 scheduled tokens 不越过预算 |
| `yield_point` | 一个 chunk 完成后的可重排位置 | 缓解长 Prefill 对 Decode 的阻塞 |

![Chunked Prefill：后缀分块与可让出执行计划](../public/02_PyTorch_Algorithms/38_chunked_prefill_schedule.svg)

### Step 3: 何时采用 PD 分池与状态交接

若 Chunked Prefill 后两类请求仍长期争用同一资源池，可以把 Prefill 与 Decode 分配给不同的逻辑池。此时需要把交接时间、链路传输、池利用率和尾延迟与吞吐一起比较，才能判断分池是否适合当前 workload。

| 决策层次 | 主要动作 | 需要同时记录 |
|---|---|---|
| 同池分块调度 | 在 chunk 边界重排 Prefill / Decode | TTFT、TPOT、Decode 抖动 |
| 逻辑 PD 分池 | 将请求分到 prefill / decode / shared 池 | 吞吐、P95、池利用率 |
| 跨池交接 | Prefill 结束后交接状态给 Decode 池 | 交接时间、传输量、失败状态 |
| 保留或调整 | 比较共享池与拆分方案 | 吞吐、尾延迟与交接代价共同达标 |

![Prefill 与 Decode 的拆分流程：分类、交接与指标比较](../public/02_PyTorch_Algorithms/38_pd_disaggregation.svg)

### Step 4：实现 CPU 调度账本与决策

题目区把本节的五个机制写成可测试函数：先识别请求压力并构造 Prefill chunk，再比较 decode-first、prefill-first 与 balanced 三种阶段预算，随后建立逻辑 PD 分池，最后比较共享池与分池的吞吐、P95 和交接预算。

| TODO | 机制责任 | 测试关注点 |
|:---|:---|:---|
| TODO 1 | 将请求分为 prefill-heavy、decode-heavy 与 mixed | 分类数量覆盖全部请求 |
| TODO 2 | 保持顺序地切分未命中 suffix | 尾块与非法 chunk size |
| TODO 3 | 在 `max_num_batched_tokens` 内生成阶段预算 | 三种策略、token 守恒与 Decode 预留 |
| TODO 4 | 让每个请求只进入一个逻辑池 | 请求不遗漏、不重复 |
| TODO 5 | 用吞吐、P95 和 handoff 预算判断是否保留 PD | 不能用单一指标决定 |


```python
from typing import Dict, List

```


```python
def summarize_request_mix(requests: List[Dict[str, int]], long_prompt_threshold: int) -> Dict[str, int]:
    """统计 prefill-heavy、decode-heavy 与 mixed 请求数量。"""
    # TODO 1（请求压力）：根据 prompt_tokens 与 decode_tokens 返回完整三类计数字典。
    # result = ???
    raise NotImplementedError


def build_chunked_prefill_plan(suffix_tokens: List[int], chunk_size: int) -> List[List[int]]:
    """把未命中 suffix 切成有序 chunk；尾块可以小于 chunk_size。"""
    # TODO 2（分块计划）：按 chunk_size 保持顺序切分 suffix；尾块可以更短。
    # chunks = ???
    raise NotImplementedError


def plan_iteration_budget(prefill_suffixes: Dict[str, int], decode_request_ids: List[str], max_num_batched_tokens: int, max_prefill_chunk_tokens: int, policy: str = 'balanced', decode_reserve_tokens: int = 1) -> Dict[str, object]:
    """在单轮 token 预算内分配 Decode token 与 Prefill chunk。"""
    # TODO 3（阶段预算）：实现 decode_first、prefill_first 与 balanced，且 used_tokens 不得越界。
    # plan = ???
    raise NotImplementedError


def plan_pd_split(requests: List[Dict[str, int]], long_prompt_threshold: int, long_decode_threshold: int) -> Dict[str, List[str]]:
    """按请求压力生成 prefill、decode 与 shared 三个逻辑池。"""
    # TODO 4（逻辑分池）：每个请求恰好进入 prefill_pool、decode_pool 或 shared_pool。
    # pools = ???
    raise NotImplementedError


def evaluate_pd_decision(baseline: Dict[str, float], split_run: Dict[str, float], max_handoff_ms: float) -> Dict[str, object]:
    """比较共享池与 PD 分池，并同时检查吞吐、P95 和交接预算。"""
    # TODO 5（分池决策）：吞吐提升、P95 不恶化且 handoff 在预算内时才保留拆分。
    # decision = ???
    raise NotImplementedError
```


```python
# 机制测试：分别检查请求分类、suffix 分块、单轮预算、逻辑分池和交接预算。
REQUESTS = [
    {'name': 'a', 'prompt_tokens': 4000, 'decode_tokens': 64},
    {'name': 'b', 'prompt_tokens': 256, 'decode_tokens': 512},
    {'name': 'c', 'prompt_tokens': 1500, 'decode_tokens': 128},
]


def test_request_mix_classification():
    """验证三类请求统计覆盖全部输入。"""
    summary = summarize_request_mix(REQUESTS, long_prompt_threshold=2048)
    assert summary == {'prefill_heavy': 1, 'decode_heavy': 1, 'mixed': 1}
    assert sum(summary.values()) == len(REQUESTS), '请求分类数量必须守恒'


def test_chunked_prefill_plan():
    """验证 suffix 顺序、尾块和非法 chunk_size。"""
    assert build_chunked_prefill_plan([9, 10, 11, 12, 13], chunk_size=2) == [[9, 10], [11, 12], [13]]
    try:
        build_chunked_prefill_plan([1], chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError('非正 chunk_size 必须拒绝')


def test_iteration_budget_policies():
    """验证三种阶段策略的成员差异与 token 预算守恒。"""
    suffixes = {'p1': 5, 'p2': 2}
    decode_ids = ['d1', 'd2']

    balanced = plan_iteration_budget(suffixes, decode_ids, 4, 2, policy='balanced', decode_reserve_tokens=1)
    assert balanced['decode_request_ids'] == ['d1']
    assert balanced['prefill_chunks'] == [{'request_id': 'p1', 'tokens': 2}, {'request_id': 'p2', 'tokens': 1}]

    decode_first = plan_iteration_budget(suffixes, decode_ids, 3, 2, policy='decode_first')
    assert decode_first['decode_request_ids'] == ['d1', 'd2']
    assert decode_first['prefill_chunks'] == [{'request_id': 'p1', 'tokens': 1}]

    prefill_first = plan_iteration_budget(suffixes, decode_ids, 3, 2, policy='prefill_first')
    assert prefill_first['decode_request_ids'] == []
    assert prefill_first['prefill_chunks'] == [{'request_id': 'p1', 'tokens': 2}, {'request_id': 'p2', 'tokens': 1}]

    for plan in (balanced, decode_first, prefill_first):
        assert plan['used_tokens'] <= plan['max_num_batched_tokens']
        assert plan['used_tokens'] + plan['remaining_tokens'] == plan['max_num_batched_tokens']


def test_iteration_budget_contract():
    """验证非法策略、非正预算与负 suffix 会被拒绝。"""
    invalid_cases = [
        ({'p': 1}, [], 0, 1, 'balanced'),
        ({'p': 1}, [], 2, 1, 'unknown'),
        ({'p': -1}, [], 2, 1, 'balanced'),
    ]
    for suffixes, decode_ids, budget, chunk, policy in invalid_cases:
        try:
            plan_iteration_budget(suffixes, decode_ids, budget, chunk, policy=policy)
        except ValueError:
            pass
        else:
            raise AssertionError('非法阶段预算输入必须拒绝')


def test_pd_pool_assignment():
    """验证每个请求只进入一个逻辑池。"""
    plan = plan_pd_split(REQUESTS, long_prompt_threshold=2048, long_decode_threshold=256)
    assert plan == {'prefill_pool': ['a'], 'decode_pool': ['b'], 'shared_pool': ['c']}


def test_pd_decision_and_handoff_budget():
    """验证吞吐、P95 与 handoff 预算共同决定是否保留拆分。"""
    baseline = {'throughput': 100, 'p95_latency_ms': 180}
    candidate = {'throughput': 126, 'p95_latency_ms': 150, 'handoff_ms': 8}
    assert evaluate_pd_decision(baseline, candidate, max_handoff_ms=10)['keep_split'] is True
    reject = evaluate_pd_decision(baseline, {**candidate, 'handoff_ms': 18}, max_handoff_ms=10)
    assert reject['keep_split'] is False, '交接超过预算时不能直接保留 PD'


def test_prefill_decode_scheduling():
    """汇总五组调度机制测试。"""
    try:
        test_request_mix_classification()
        test_chunked_prefill_plan()
        test_iteration_budget_policies()
        test_iteration_budget_contract()
        test_pd_pool_assignment()
        test_pd_decision_and_handoff_budget()
        print('✅ Prefill/Decode 调度机制测试通过：分类、分块、阶段预算、分池和决策均通过。')
    except NotImplementedError:
        raise
    except (NameError, AttributeError, TypeError, ValueError, AssertionError) as error:
        raise NotImplementedError('请先完成 TODO 代码或检查字段名！') from error


test_prefill_decode_scheduling()

```

## 参考代码与解析

### 代码


```python
def summarize_request_mix(requests: List[Dict[str, int]], long_prompt_threshold: int) -> Dict[str, int]:
    """统计 prefill-heavy、decode-heavy 与 mixed 请求数量。"""
    # TODO 1: 按请求的输入和生成长度完成三类统计
    result = {'prefill_heavy': 0, 'decode_heavy': 0, 'mixed': 0}
    for request in requests:
        prompt_tokens = request.get('prompt_tokens', 0)
        decode_tokens = request.get('decode_tokens', 0)
        if prompt_tokens > long_prompt_threshold and decode_tokens <= long_prompt_threshold // 8:
            result['prefill_heavy'] += 1
        elif decode_tokens > long_prompt_threshold // 8 and prompt_tokens <= long_prompt_threshold:
            result['decode_heavy'] += 1
        else:
            result['mixed'] += 1
    return result


def build_chunked_prefill_plan(suffix_tokens: List[int], chunk_size: int) -> List[List[int]]:
    """把未命中 suffix 切成有序 chunk；尾块可以小于 chunk_size。"""
    # TODO 2: 生成 Chunked Prefill 的执行计划
    if chunk_size <= 0:
        raise ValueError('chunk_size 必须为正数')
    suffix = list(suffix_tokens)
    return [suffix[start:start + chunk_size] for start in range(0, len(suffix), chunk_size)]


def plan_iteration_budget(prefill_suffixes: Dict[str, int], decode_request_ids: List[str], max_num_batched_tokens: int, max_prefill_chunk_tokens: int, policy: str = 'balanced', decode_reserve_tokens: int = 1) -> Dict[str, object]:
    """在单轮 token 预算内分配 Decode token 与 Prefill chunk。"""
    # TODO 3: 用阶段优先级分配预算，并保持 used_tokens 不超过单轮上限
    if max_num_batched_tokens <= 0 or max_prefill_chunk_tokens <= 0:
        raise ValueError('token 预算与 Prefill chunk 上限必须为正数')
    if decode_reserve_tokens < 0:
        raise ValueError('decode_reserve_tokens 不能为负数')
    if policy not in {'decode_first', 'prefill_first', 'balanced'}:
        raise ValueError(f'不支持的阶段策略: {policy}')
    if any(tokens < 0 for tokens in prefill_suffixes.values()):
        raise ValueError('Prefill suffix 长度不能为负数')
    if len(set(decode_request_ids)) != len(decode_request_ids):
        raise ValueError('Decode 请求 ID 不能重复')

    remaining = max_num_batched_tokens
    scheduled_decode: List[str] = []
    prefill_chunks: List[Dict[str, object]] = []

    def schedule_decode(limit: int | None = None) -> None:
        nonlocal remaining
        pending = [request_id for request_id in decode_request_ids if request_id not in scheduled_decode]
        take = min(len(pending), remaining) if limit is None else min(len(pending), remaining, limit)
        scheduled_decode.extend(pending[:take])
        remaining -= take

    def schedule_prefill() -> None:
        nonlocal remaining
        for request_id, suffix_tokens in prefill_suffixes.items():
            if remaining == 0:
                break
            chunk_tokens = min(suffix_tokens, max_prefill_chunk_tokens, remaining)
            if chunk_tokens > 0:
                prefill_chunks.append({'request_id': request_id, 'tokens': chunk_tokens})
                remaining -= chunk_tokens

    if policy == 'decode_first':
        schedule_decode()
        schedule_prefill()
    elif policy == 'prefill_first':
        schedule_prefill()
        schedule_decode()
    else:
        schedule_decode(limit=decode_reserve_tokens)
        schedule_prefill()
        schedule_decode()

    used_tokens = max_num_batched_tokens - remaining
    return {
        'policy': policy,
        'max_num_batched_tokens': max_num_batched_tokens,
        'decode_request_ids': scheduled_decode,
        'prefill_chunks': prefill_chunks,
        'used_tokens': used_tokens,
        'remaining_tokens': remaining,
    }


def plan_pd_split(requests: List[Dict[str, int]], long_prompt_threshold: int, long_decode_threshold: int) -> Dict[str, List[str]]:
    """按请求压力生成 prefill、decode 与 shared 三个逻辑池。"""
    # TODO 4: 每个请求只进入一个逻辑池
    prefill_pool, decode_pool, shared_pool = [], [], []
    for request in requests:
        name = request.get('name', 'request')
        if request.get('prompt_tokens', 0) > long_prompt_threshold and request.get('decode_tokens', 0) <= long_decode_threshold:
            prefill_pool.append(name)
        elif request.get('decode_tokens', 0) > long_decode_threshold and request.get('prompt_tokens', 0) <= long_prompt_threshold:
            decode_pool.append(name)
        else:
            shared_pool.append(name)
    return {'prefill_pool': prefill_pool, 'decode_pool': decode_pool, 'shared_pool': shared_pool}


def evaluate_pd_decision(baseline: Dict[str, float], split_run: Dict[str, float], max_handoff_ms: float) -> Dict[str, object]:
    """比较共享池与 PD 分池，并同时检查吞吐、P95 和交接预算。"""
    # TODO 5: 输出是否保留当前 PD 方案
    throughput_gain = split_run.get('throughput', 0.0) - baseline.get('throughput', 0.0)
    latency_delta_ms = split_run.get('p95_latency_ms', 0.0) - baseline.get('p95_latency_ms', 0.0)
    handoff_ms = split_run.get('handoff_ms', 0.0)
    handoff_ok = handoff_ms <= max_handoff_ms
    keep_split = throughput_gain > 0 and latency_delta_ms <= 0 and handoff_ok
    return {'throughput_gain': throughput_gain, 'latency_delta_ms': latency_delta_ms, 'handoff_ms': handoff_ms, 'keep_split': keep_split}
```

### 解析

**TODO 1：识别请求压力**

- 统计结果必须覆盖全部请求；它把输入长度和生成长度变成调度器可以使用的请求画像。

**TODO 2：生成 Chunked Prefill 计划**

- 只切分 Prefix Cache 未命中的 suffix，不改变 token 顺序；每个 chunk 结束后，调度器才有机会重新安排 Decode 与后续 Prefill。

**TODO 3：分配单轮阶段预算**

- `decode_first` 优先保护所有可容纳的 Decode 请求，`prefill_first` 优先消费 Prefill chunk，`balanced` 先保留少量 Decode token，再让 Prefill 使用剩余预算。三种策略都必须满足 `used_tokens <= max_num_batched_tokens`。

**TODO 4：建立逻辑 PD 分池**

- 每个请求只进入 prefill、decode 或 shared 之一。这个账本帮助检查“谁在争用什么资源”，也避免重复服务。

**TODO 5：判断是否保留分池**

- 分池必须同时带来吞吐改善、P95 不恶化，并让 handoff 保持在预算内；任何一项失败，都应继续调整而不是直接保留。
### 真实 GPU / backend 验证出口

本节 Step 4 验证请求分类、Chunked Prefill、逻辑分池与交接预算。环境预检只能说明“能否启动”，不能证明 Chunked Prefill 或 PD 分离有收益，因此本节不保留 伪 GPU 证据或预检结果表。

真实对照需要固定模型、prompt/decode 分布、并发度、Token 预算和 worker 资源，比较 shared pool、Chunked Prefill 与 PD split，同时记录 TTFT、TPOT、吞吐、P95/P99、池利用率和状态交接时间。该证据由 [70 Serving 调度基准](./70_Serving_Scheduler_Benchmark.md) 与 [81 分布式推理项目](./81_Distributed_Inference_Project.md) 承接。
## 相关阅读

完成 Chunked Prefill、分池与交接预算判断后，可以继续阅读异构 PD、真实服务调度和多实例验证。

- [DistServe 原论文：Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [SGLang PD Disaggregation 文档](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)
- [39. Hetero PD and Serving Tiers | 异构 PD 与服务分层](./39_Hetero_PD_and_Serving_Tiers.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)