# 36. Decode Scheduling | 解码调度
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `解码`, `Scheduling` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面几节已经看过投机解码和多 Token 生成。现在把视角从单个请求扩展到一组同时生成的请求：有的刚完成输入处理，有的正在逐 Token 生成，有的即将结束。调度器需要根据这些状态决定下一步推进谁。

本节关注请求级选择：候选请求采用不同排序规则时，服务顺序、等待时间和生成进度会如何变化；当高优先级或临近 deadline 的请求无法准入时，还要判断能否安全抢占并在之后恢复。每轮批次成员如何动态加入和退出，继续由 Continuous Batching 机制处理。

**关键词：** `decode scheduling`, `admission`, `aging`, `fairness`, `preemption`

---

## 前置阅读

**导语：** 进入本节前，先能区分请求的 prefill、decode 和完成状态，再观察调度器如何依据状态选择下一步请求。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)

---

### Step 1: 多请求为什么需要解码调度

把多个请求放进同一个等待队列后，它们会竞争下一轮 Decode 的执行机会。本节用请求阶段、缓存命中和业务优先级确定谁先获得执行机会；这些选择会改变生成进度、响应延迟和完成顺序。

固定优先级策略直接使用业务优先级；aging 把等待步数折算成优先级增量，避免低优先级请求长期饥饿；deadline 策略进一步比较剩余步数与完成预算。调度前还要检查 KV block 是否足够：容量不足时只能排队，或抢占可安全暂停且优先级更低的已准入请求。

| 字段或观察对象 | 在本节中的角色 | 对当前选择的影响 |
| --- | --- | --- |
| 活动请求集合 | 收集尚未完成、可继续推进的请求 | 形成可重排的候选集合 |
| 阶段、缓存命中、业务优先级 | 排序键的主要决策字段 | 决定谁先获得一个执行步 |
| 总长度、请求编号 | 稳定 tie-breaker | 只在前面字段相同时决定顺序 |
| `wait_steps` | 等待与潜在饥饿的观测字段 | 固定策略只记录；aging 策略将其折算为优先级增量 |
| `required_blocks` / 可用 KV block | 请求的容量需求与系统容量状态 | 决定请求当前能否准入；不负责 block 的物理分配 |
| `deadline_step` / `preemptible` | 截止步与能否安全暂停 | 决定紧急程度，以及容量不足时是否允许抢占 |
| 完成进度与事件序列 | 调度后的结果记录 | 用于解释完成顺序、等待与公平性 |

![Decode 调度概念图：请求状态进入调度器并循环推进](../public/02_PyTorch_Algorithms/36_decode_scheduling_board.svg)

### Step 2: 请求状态与生命周期机制

调度前先把请求看成会变化的状态对象。正常路径是 waiting → admitted → prefill → decode → finished；容量压力下，已准入请求还可能进入 preempted，释放其 KV 预算，之后重新准入并从保存的阶段继续。

| 机制环节 | 状态或动作 | 机制含义 |
|---|---|---|
| 状态对象 | 输入、阶段、生成进度、生成上限和缓存信息 | 说明请求当前处于什么阶段，还需要生成多少内容 |
| 状态转移 | waiting → admitted → prefill → decode | 请求获得容量后完成输入处理并逐步生成 |
| 抢占与恢复 | admitted → preempted → admitted | 保存阶段和生成进度，释放后重新申请容量 |
| 活动集合 | 尚未完成的请求 | 只保留仍需要推进的请求，已完成请求退出队列 |
| 单步推进 | 选中一个请求前进一轮 | 每一轮只推进当前选中的请求，其他请求继续等待 |
| 完成条件 | 达到生成上限 | 请求完成并从活动集合中退出 |

### Step 3: 调度策略与排序优先级

有了请求生命周期，还需要一条可重复的选择规则来比较活动请求。本节固定排序键为“阶段 → 缓存命中 → 业务优先级 → 总长度 → 请求编号”：字段越靠前，越先决定结果；排序键越小的请求越早获得推进机会。业务优先级采用负号转换，因此数值越大的请求越靠前。

三条策略共享阶段、缓存命中和稳定 tie-breaker：固定策略使用业务优先级，aging 使用 `priority + wait_steps // aging_interval`，deadline 策略优先选择完成余量更小的请求。若最紧急请求容量不足，调度器只能抢占 `preemptible=True` 且优先级更低的已准入请求；抢占必须保留原阶段和生成进度，恢复后不能重新开始。

![Decode 调度排序键：从左到右比较字段](../public/02_PyTorch_Algorithms/36_decode_sort_key.svg)

### Step 4: 实现请求状态与调度循环

请补全下方 `DecodeSchedulerSim`，跑通“入队 → 固定/aging/deadline 排序 → 容量准入或抢占 → 单步推进 → 恢复或释放容量”这条链路。物理 block 管理由缓存机制负责，本题只模拟调度器消费和归还容量状态。

| TODO / 实现对象 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| TODO 1 / `enqueue` | 构造请求状态并加入队列 | 记录容量、deadline 和可抢占性 | 输入状态契约 |
| TODO 2 / `_schedule_key` | 比较 fixed、aging 与 deadline | deadline 只在显式策略下改变顺序；tie-breaker 稳定 | 公平性与紧急度对照 |
| TODO 3 / `_ensure_capacity` | 执行准入或选择抢占对象 | 只能抢占可暂停且优先级更低的请求；状态与进度必须保留 | 容量与抢占不变量 |
| TODO 4 / `step` | 推进并记录恢复、完成和 deadline | 恢复不重置进度，完成释放容量，未选请求累计等待 | 状态、容量、SLO 与事件观测 |
| `run` / 事件记录 | 使用已有循环推进并保存事件 | `max_steps` 只作循环保护 | 事件数量与异常输入 |

```python
from dataclasses import dataclass
from typing import Dict, List, Literal

```


```python
Phase = Literal['prefill', 'decode']
RequestStatus = Literal['waiting', 'running', 'preempted', 'finished']


@dataclass
class RequestState:
    """记录一个请求在 prefill/decode 调度过程中的最小状态。"""
    request_id: int
    prompt_len: int
    max_new_tokens: int
    generated_len: int = 0
    phase: Phase = 'prefill'
    priority: int = 0
    cache_hit: bool = False
    wait_steps: int = 0
    required_blocks: int = 1
    admitted: bool = False
    deadline_step: int | None = None
    preemptible: bool = True
    status: RequestStatus = 'waiting'
    preemptions: int = 0

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def done(self) -> bool:
        return self.generated_len >= self.max_new_tokens


class DecodeSchedulerSim:
    """用事件序列观察多个请求如何在 prefill 与 decode 之间推进。"""

    def __init__(self, policy: str = 'fixed_priority', free_kv_blocks: int = 1_000_000, aging_interval: int = 4):
        if policy not in {'fixed_priority', 'aging', 'deadline'}:
            raise ValueError('policy 必须是 fixed_priority、aging 或 deadline')
        if free_kv_blocks < 0 or aging_interval <= 0:
            raise ValueError('容量不能为负，aging_interval 必须为正数')
        self.policy = policy
        self.free_kv_blocks = free_kv_blocks
        self.aging_interval = aging_interval
        self.clock = 0
        self.queue: List[RequestState] = []
        self.timeline: List[Dict[str, int | str]] = []

    def enqueue(self, request_id: int, prompt_len: int, max_new_tokens: int = 2, priority: int = 0, cache_hit: bool = False, required_blocks: int = 1, deadline_step: int | None = None, preemptible: bool = True) -> None:
        """校验并登记请求；请求从 prefill 阶段开始。"""
        if request_id in {item.request_id for item in self.queue}:
            raise ValueError(f'重复的 request_id: {request_id}')
        if prompt_len <= 0 or max_new_tokens <= 0 or required_blocks <= 0:
            raise ValueError('prompt_len、max_new_tokens 和 required_blocks 必须为正数')
        if deadline_step is not None and deadline_step <= self.clock:
            raise ValueError('deadline_step 必须晚于当前调度时钟')
        # ==========================================
        # TODO 1: 构造请求状态，并加入调度队列
        # 提示：保留优先级、容量、deadline 与 preemptible；状态从 waiting 开始。
        # request：待加入队列的 RequestState，保存请求的最小调度状态
        # ==========================================
        # request = ???
        self.queue.append(request)

    def _effective_priority(self, req: RequestState) -> int:
        """返回 fixed/aging 共用的有效业务优先级。"""
        return req.priority + (req.wait_steps // self.aging_interval if self.policy == 'aging' else 0)

    def _remaining_steps(self, req: RequestState) -> int:
        return (1 if req.phase == 'prefill' else 0) + max(req.max_new_tokens - req.generated_len, 0)

    def _schedule_key(self, req: RequestState) -> tuple[int, int, int, int, int, int]:
        """返回 deadline、阶段、缓存、有效优先级与稳定字段组成的排序键。"""
        # TODO 2（排序策略）：deadline 模式使用完成余量；其他模式不启用 deadline 排序。
        # deadline_slack = ???
        phase_rank = 0 if req.phase == 'prefill' else 1
        cache_rank = 0 if req.cache_hit else 1
        return (deadline_slack, phase_rank, cache_rank, -self._effective_priority(req), req.total_len, req.request_id)

    def _ensure_capacity(self, challenger: RequestState) -> int | None:
        """为 challenger 准入；必要时抢占一个低优先级、可暂停的已准入请求。"""
        # TODO 3（准入与抢占）：返回被抢占的 request_id；无需抢占时返回 None。
        # 约束：只有 challenger 更紧急时才可抢占，且 victim 的阶段与 generated_len 不得重置。
        raise NotImplementedError

    def step(self) -> Dict[str, int | str] | None:
        """选择并推进一个活动请求，同时记录事件和等待步数。"""
        candidates = [req for req in self.queue if not req.done]
        if not candidates:
            return None

        # ==========================================
        # TODO 4: 按排序键选择请求，并通过 _ensure_capacity 完成准入、抢占或恢复。
        # chosen：本轮实际获得一个调度步的 RequestState
        # ==========================================
        # chosen = ???
        # preempted_request_id = ???

        for req in candidates:
            if req is not chosen:
                req.wait_steps += 1

        if chosen.phase == 'prefill':
            if not chosen.admitted:
                self.free_kv_blocks -= chosen.required_blocks
                chosen.admitted = True
            chosen.status = 'running'
            chosen.phase = 'decode'
            action = 'prefill_to_decode'
        else:
            chosen.generated_len += 1
            action = 'finish' if chosen.done else 'decode_one_step'
            if chosen.done and chosen.admitted:
                self.free_kv_blocks += chosen.required_blocks
                chosen.admitted = False
                chosen.status = 'finished'

        event = {
            'request_id': chosen.request_id,
            'phase': 'prefill' if action == 'prefill_to_decode' else 'decode',
            'action': action,
            'prompt_len': chosen.prompt_len,
            'generated_len': chosen.generated_len,
            'max_new_tokens': chosen.max_new_tokens,
            'wait_steps': chosen.wait_steps,
            'free_kv_blocks': self.free_kv_blocks,
            'preempted_request_id': preempted_request_id,
            'deadline_missed': chosen.deadline_step is not None and self.clock + 1 + self._remaining_steps(chosen) > chosen.deadline_step,
        }
        self.timeline.append(event)
        self.clock += 1
        return event

    def run(self, max_steps: int = 100) -> List[Dict[str, int | str]]:
        """持续推进请求，直到全部完成或达到步数上限。"""
        if max_steps < 0:
            raise ValueError('max_steps 不能为负数')
        steps = 0
        while steps < max_steps:
            event = self.step()
            if event is None:
                break
            # 每个成功事件消耗一个调度步；这是 max_steps 的防护计数，不是调度策略。
            steps = steps + 1
        return self.timeline

```


```python
# 机制测试：分别验证请求状态、排序、公平性、容量准入和输入保护。
def test_enqueue_contract():
    """验证新请求从 prefill 状态入队，并保留输入契约字段。"""
    sim = DecodeSchedulerSim()
    sim.enqueue(request_id=1, prompt_len=2, max_new_tokens=2, priority=2, cache_hit=True)
    req = sim.queue[0]
    assert req.phase == 'prefill' and req.generated_len == 0
    assert req.prompt_len == 2 and req.max_new_tokens == 2
    assert req.required_blocks == 1 and not req.admitted


def test_schedule_key_decision_order():
    """验证阶段、缓存命中、业务优先级与稳定字段的既定顺序。"""
    sim = DecodeSchedulerSim()
    sim.enqueue(1, 4, 1, priority=0, cache_hit=False)
    sim.enqueue(2, 4, 1, priority=9, cache_hit=True)
    sim.queue[1].phase = 'decode'
    # phase 比缓存和 priority 更靠前，因此 prefill 请求先被选中。
    assert sim._schedule_key(sim.queue[0]) < sim._schedule_key(sim.queue[1])

    sim = DecodeSchedulerSim()
    sim.enqueue(1, 4, 1, priority=1, cache_hit=False)
    sim.enqueue(2, 4, 1, priority=1, cache_hit=True)
    assert sim._schedule_key(sim.queue[1]) < sim._schedule_key(sim.queue[0])

    sim = DecodeSchedulerSim()
    sim.enqueue(1, 4, 1, priority=1, cache_hit=True)
    sim.enqueue(2, 4, 1, priority=3, cache_hit=True)
    assert sim._schedule_key(sim.queue[1]) < sim._schedule_key(sim.queue[0])
    assert sim.step()['request_id'] == 2


def test_wait_steps_is_observation_not_priority():
    """验证等待步数只记录公平性，不得改变当前排序键。"""
    sim = DecodeSchedulerSim()
    sim.enqueue(1, 4, 1, priority=1, cache_hit=True)
    sim.enqueue(2, 4, 1, priority=1, cache_hit=True)
    first, second = sim.queue
    key_before = sim._schedule_key(first)
    first.wait_steps = 50
    second.wait_steps = 0
    assert sim._schedule_key(first) == key_before
    assert sim._schedule_key(first) < sim._schedule_key(second)


def test_aging_changes_effective_priority():
    """验证 aging 只在显式启用时提高等待请求的有效优先级。"""
    sim = DecodeSchedulerSim(policy='aging', aging_interval=2)
    sim.enqueue(1, 4, 1, priority=1, cache_hit=True)
    sim.enqueue(2, 4, 1, priority=3, cache_hit=True)
    sim.queue[0].wait_steps = 6
    assert sim._effective_priority(sim.queue[0]) == 4
    assert sim._schedule_key(sim.queue[0]) < sim._schedule_key(sim.queue[1])


def test_capacity_admission_and_release():
    """验证容量不足的请求暂不准入，完成请求会释放 KV block。"""
    sim = DecodeSchedulerSim(free_kv_blocks=2)
    sim.enqueue(1, 4, 1, priority=9, required_blocks=3)
    sim.enqueue(2, 4, 1, priority=1, required_blocks=2)
    assert sim.step()['request_id'] == 2
    assert sim.free_kv_blocks == 0
    assert sim.step()['action'] == 'finish'
    assert sim.free_kv_blocks == 2


def test_preemption_and_resume_preserve_progress():
    """验证高优先级请求可抢占，victim 恢复时保留阶段与生成进度。"""
    sim = DecodeSchedulerSim(free_kv_blocks=2)
    sim.enqueue(1, 4, 2, priority=1, required_blocks=2, preemptible=True)
    assert sim.step()['request_id'] == 1
    victim = sim.queue[0]
    assert victim.phase == 'decode' and victim.admitted
    sim.enqueue(2, 2, 1, priority=9, required_blocks=2)
    event = sim.step()
    assert event['request_id'] == 2 and event['preempted_request_id'] == 1
    assert victim.status == 'preempted' and victim.phase == 'decode' and victim.generated_len == 0
    sim.run(max_steps=10)
    assert victim.done and victim.preemptions == 1


def test_non_preemptible_and_deadline_policy():
    """验证不可抢占请求受保护，并验证 deadline 余量进入排序与违约记录。"""
    sim = DecodeSchedulerSim(free_kv_blocks=2)
    sim.enqueue(1, 4, 2, priority=1, required_blocks=2, preemptible=False)
    sim.step()
    sim.enqueue(2, 2, 1, priority=9, required_blocks=2)
    event = sim.step()
    assert event['request_id'] == 1 and event['preempted_request_id'] is None

    deadline_sim = DecodeSchedulerSim(policy='deadline')
    deadline_sim.enqueue(1, 4, 4, priority=9, deadline_step=20)
    deadline_sim.enqueue(2, 2, 1, priority=0, deadline_step=1)
    event = deadline_sim.step()
    assert event['request_id'] == 2 and event['deadline_missed'] is True


def test_state_progression_and_completion():
    """验证 prefill 到 decode、完成退出和等待累计。"""
    sim = DecodeSchedulerSim()
    sim.enqueue(1, 2, 2, priority=2, cache_hit=True)
    sim.enqueue(2, 3, 3, priority=1, cache_hit=False)
    events = sim.run(max_steps=20)
    assert len(events) == 7
    assert any(event['action'] == 'finish' for event in events)
    assert all(req.done for req in sim.queue)
    assert any(event['wait_steps'] > 0 for event in events)


def test_input_guards_and_step_limit():
    """验证非法请求与调度步数上限的保护。"""
    sim = DecodeSchedulerSim()
    for kwargs in ({'request_id': 1, 'prompt_len': 0}, {'request_id': 1, 'prompt_len': 1, 'max_new_tokens': 0}, {'request_id': 1, 'prompt_len': 1, 'required_blocks': 0}):
        try:
            sim.enqueue(**kwargs)
        except ValueError:
            continue
        raise AssertionError('非法请求参数应被拒绝')
    sim.enqueue(1, 1, 3)
    assert len(sim.run(max_steps=2)) == 2
    try:
        sim.run(max_steps=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('负数 max_steps 应被拒绝')


def test_decode_scheduler():
    test_enqueue_contract()
    test_schedule_key_decision_order()
    test_wait_steps_is_observation_not_priority()
    test_aging_changes_effective_priority()
    test_capacity_admission_and_release()
    test_preemption_and_resume_preserve_progress()
    test_non_preemptible_and_deadline_policy()
    test_state_progression_and_completion()
    test_input_guards_and_step_limit()
    print('✅ DecodeSchedulerSim 机制测试通过：排序、公平性、准入、抢占、恢复和 deadline 均通过。')


test_decode_scheduler()
```


## 参考代码与解析

### 代码

```python
# TODO：下面是题目区的参考实现。

Phase = Literal['prefill', 'decode']
RequestStatus = Literal['waiting', 'running', 'preempted', 'finished']


@dataclass
class RequestState:
    """记录一个请求在 prefill/decode 调度过程中的最小状态。"""
    request_id: int
    prompt_len: int
    max_new_tokens: int
    generated_len: int = 0
    phase: Phase = 'prefill'
    priority: int = 0
    cache_hit: bool = False
    wait_steps: int = 0
    required_blocks: int = 1
    admitted: bool = False
    deadline_step: int | None = None
    preemptible: bool = True
    status: RequestStatus = 'waiting'
    preemptions: int = 0

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def done(self) -> bool:
        return self.generated_len >= self.max_new_tokens


class DecodeSchedulerSim:
    """用事件序列观察多个请求如何在 prefill 与 decode 之间推进。"""

    def __init__(self, policy: str = 'fixed_priority', free_kv_blocks: int = 1_000_000, aging_interval: int = 4):
        if policy not in {'fixed_priority', 'aging', 'deadline'}:
            raise ValueError('policy 必须是 fixed_priority、aging 或 deadline')
        if free_kv_blocks < 0 or aging_interval <= 0:
            raise ValueError('容量不能为负，aging_interval 必须为正数')
        self.policy = policy
        self.free_kv_blocks = free_kv_blocks
        self.aging_interval = aging_interval
        self.clock = 0
        self.queue: List[RequestState] = []
        self.timeline: List[Dict[str, int | str]] = []

    def enqueue(self, request_id: int, prompt_len: int, max_new_tokens: int = 2, priority: int = 0, cache_hit: bool = False, required_blocks: int = 1, deadline_step: int | None = None, preemptible: bool = True) -> None:
        """校验并登记请求；请求从 prefill 阶段开始。"""
        if request_id in {item.request_id for item in self.queue}:
            raise ValueError(f'重复的 request_id: {request_id}')
        if prompt_len <= 0 or max_new_tokens <= 0 or required_blocks <= 0:
            raise ValueError('prompt_len、max_new_tokens 和 required_blocks 必须为正数')
        if deadline_step is not None and deadline_step <= self.clock:
            raise ValueError('deadline_step 必须晚于当前调度时钟')
        # ==========================================
        # TODO 1: 构造请求状态，并加入调度队列
        # 提示: RequestState 默认处于 prefill 阶段；分别传入输入长度和生成上限
        # ==========================================
        request = RequestState(
            request_id=request_id,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            priority=priority,
            cache_hit=cache_hit,
            required_blocks=required_blocks,
            deadline_step=deadline_step,
            preemptible=preemptible,
        )
        self.queue.append(request)

    def _effective_priority(self, req: RequestState) -> int:
        """返回固定策略或 aging 策略使用的有效优先级。"""
        # ==========================================
        # TODO 2: 计算固定优先级或 aging 策略的有效优先级
        # ==========================================
        effective_priority = req.priority
        if self.policy == 'aging':
            effective_priority += req.wait_steps // self.aging_interval
        return effective_priority

    def _remaining_steps(self, req: RequestState) -> int:
        return (1 if req.phase == 'prefill' else 0) + max(req.max_new_tokens - req.generated_len, 0)

    def _schedule_key(self, req: RequestState) -> tuple[int, int, int, int, int, int]:
        """返回 deadline、阶段、缓存、有效优先级与稳定字段组成的排序键。"""
        # TODO 2: deadline 模式优先比较完成余量。
        deadline_slack = 1_000_000_000
        if self.policy == 'deadline' and req.deadline_step is not None:
            deadline_slack = req.deadline_step - self.clock - self._remaining_steps(req)
        phase_rank = 0 if req.phase == 'prefill' else 1
        cache_rank = 0 if req.cache_hit else 1
        return (deadline_slack, phase_rank, cache_rank, -self._effective_priority(req), req.total_len, req.request_id)

    def _ensure_capacity(self, challenger: RequestState) -> int | None:
        """为 challenger 准入，必要时抢占一个低优先级、可暂停的请求。"""
        # TODO 3: 已准入请求直接继续；空闲容量不足时寻找一个可安全抢占的 victim。
        if challenger.admitted:
            challenger.status = 'running'
            return None
        if challenger.required_blocks <= self.free_kv_blocks:
            self.free_kv_blocks -= challenger.required_blocks
            challenger.admitted = True
            challenger.status = 'running'
            return None
        victims = [req for req in self.queue if req.admitted and not req.done and req.preemptible and self._effective_priority(req) < self._effective_priority(challenger)]
        victims.sort(key=lambda req: (self._effective_priority(req), -req.required_blocks, req.request_id))
        victim = next((req for req in victims if self.free_kv_blocks + req.required_blocks >= challenger.required_blocks), None)
        if victim is None:
            return None
        victim.admitted = False
        victim.status = 'preempted'
        victim.preemptions += 1
        self.free_kv_blocks += victim.required_blocks
        self.free_kv_blocks -= challenger.required_blocks
        challenger.admitted = True
        challenger.status = 'running'
        return victim.request_id

    def step(self) -> Dict[str, int | str] | None:
        """选择并推进一个活动请求，同时记录事件和等待步数。"""
        candidates = [req for req in self.queue if not req.done]
        if not candidates:
            return None

        # ==========================================
        # TODO 4: 按排序键尝试准入；容量不足时允许 _ensure_capacity 执行安全抢占。
        # ==========================================
        chosen = None
        preempted_request_id = None
        for candidate in sorted(candidates, key=self._schedule_key):
            preempted_request_id = self._ensure_capacity(candidate)
            if candidate.admitted:
                chosen = candidate
                break
        if chosen is None:
            return None
        for req in candidates:
            if req is not chosen:
                req.wait_steps += 1
        if chosen.phase == 'prefill':
            chosen.phase = 'decode'
            action = 'prefill_to_decode'
        else:
            chosen.generated_len += 1
            action = 'finish' if chosen.done else 'decode_one_step'
            if chosen.done and chosen.admitted:
                self.free_kv_blocks += chosen.required_blocks
                chosen.admitted = False
                chosen.status = 'finished'

        event = {
            'request_id': chosen.request_id,
            'phase': 'prefill' if action == 'prefill_to_decode' else 'decode',
            'action': action,
            'prompt_len': chosen.prompt_len,
            'generated_len': chosen.generated_len,
            'max_new_tokens': chosen.max_new_tokens,
            'wait_steps': chosen.wait_steps,
            'free_kv_blocks': self.free_kv_blocks,
            'preempted_request_id': preempted_request_id,
            'deadline_missed': chosen.deadline_step is not None and self.clock + 1 + self._remaining_steps(chosen) > chosen.deadline_step,
        }
        self.timeline.append(event)
        self.clock += 1
        return event

    def run(self, max_steps: int = 100) -> List[Dict[str, int | str]]:
        """持续推进请求，直到全部完成或达到步数上限。"""
        if max_steps < 0:
            raise ValueError('max_steps 不能为负数')
        steps = 0
        while steps < max_steps:
            event = self.step()
            if event is None:
                break
            # 每个成功事件消耗一个调度步；这是 max_steps 的防护计数，不是调度策略。
            steps = steps + 1
        return self.timeline

```

### 解析

本题依次实现请求状态登记、固定/aging/deadline 排序、容量准入、安全抢占和恢复。答案代码保留与题目区相同的状态流，只补全每处机制决策。

**TODO 1：构造请求状态**

- 使用 `RequestState` 保存输入长度、生成上限、阶段、优先级、KV block、deadline 和可抢占性。
- 新请求默认处于 `prefill`，`generated_len` 从 0 开始，再加入调度队列。

**TODO 2：计算排序键**

- 固定策略直接使用业务优先级，aging 按等待步数提高有效优先级，deadline 策略先比较预计完成余量。
- 非 deadline 策略保持既有阶段、cache 命中和业务优先级顺序；长度和请求编号只负责稳定打破平局。

**TODO 3：执行容量准入与安全抢占**

- 空闲容量足够时直接准入；否则只能选择 `preemptible=True`、优先级更低且释放后足够容纳 challenger 的 victim。
- 抢占只改变 admitted/status 和容量账本，不重置 victim 的 phase、generated_len 或 deadline。

**TODO 4：选择并推进请求**

- 按排序键依次尝试准入；成功时推进请求，恢复被抢占请求时从保存阶段继续，完成时释放容量。
- 未被选中的请求增加 `wait_steps`；事件同时记录抢占对象和预计 deadline 是否已经无法满足。

**测试如何定位问题**

- 输入契约测试检查非法长度、deadline、block 需求和重复 `request_id`；策略测试分别检查 fixed、aging 与 deadline。
- 容量测试检查 admission/release；抢占测试检查不可抢占保护与恢复；deadline 测试检查紧急排序和违约记录。
### 真实 GPU 验证出口

本节 Step 4 比较 FCFS、业务优先级、aging、SLO 与容量准入的请求选择结果。随机矩阵微基准不能反映队列等待、饥饿、抢占或尾延迟，因此不作为 GPU 证据。

真实调度实验应在同一 Serving backend 上固定模型、到达轨迹、输入/输出长度和 KV 预算，仅替换请求选择策略，并记录 queue wait、TTFT、TPOT、P95/P99、饥饿与拒绝请求。该对照由 [70 Serving 调度基准](./70_Serving_Scheduler_Benchmark.md) 承接。
## 相关阅读

完成请求状态、选择规则和推进顺序的模拟后，可以继续阅读推理服务调度的论文与真实基准项目。

- [Orca 原论文：A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [37. KV Cache Lifecycle and Capacity Control | KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)
- [38. Prefill/Decode Scheduling | Prefill/Decode 调度](./38_Prefill_Decode_Scheduling.md)
- [70. Serving Scheduler Benchmark | 服务调度基准项目](./70_Serving_Scheduler_Benchmark.md)
