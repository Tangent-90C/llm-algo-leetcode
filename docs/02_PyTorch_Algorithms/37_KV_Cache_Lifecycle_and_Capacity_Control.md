# 37. KV Cache Lifecycle and Capacity Control | KV Cache 生命周期与容量控制
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `KV Cache`, `生命周期` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

KV Cache 会随 Prefill 分配、随 Decode 追加，也可能被多个请求共享；请求结束、容量不足或状态迁移时，还会发生引用释放、驱逐、抢占与回收。只有把这些事件放进同一生命周期，调度器才能获得可信的可用容量。

本节沿着“allocate → append → share / reference → capacity pressure → evict / preempt → reclaim”的顺序建立容量账本。学习后，你应能解释请求在什么时刻占用或释放 block，以及为什么缓存容量状态应由本节提供、请求先后顺序则由 Serving 调度器决定。


**关键词：** `KV Cache 生命周期`、`引用计数`、`容量控制`、`驱逐与回收`

---

## 前置阅读

进入本节前，先理解分页 block 和前缀复用如何改变物理存储；本节再把它们放进完整生命周期。
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [24. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./24_Prefix_Cache_Matching_and_Reuse.md)

---

### Step 1：KV Cache 在请求生命周期中如何变化

Prefill 为请求建立初始缓存，Decode 持续向尾块追加 token；前缀命中时，新请求引用已有 block，而不是复制同一份状态。请求结束只会减少引用，引用归零后，缓存才可能保留为可复用状态，或被驱逐、回收。

| 生命周期事件 | 状态变化 | 容量账本变化 |
|---|---|---|
| allocate | 为新前缀取得首批 block | 已分配 block 增加 |
| append | 向尾块写入 token，跨越边界时申请新 block | 可能只增加有效 token，也可能增加 block |
| share / reference | 另一个请求复用相同前缀 | 引用数增加，物理 block 不重复增加 |
| release | 请求结束或取消 | 引用数减少，不一定立即释放 block |
| evict / reclaim | 淘汰未引用缓存或主动释放状态 | 已分配 block 减少，可用容量增加 |

![KV Cache 生命周期、缓存状态与容量决策](../public/02_PyTorch_Algorithms/37_kv_cache_scheduling_map.svg)

### Step 2：引用、尾块与容量账本

容量判断不能只看“缓存条目有几个”。系统还要区分有效 token、尾块余量、物理 block 和活动引用：尾块尚有空间时，append 不增加物理占用；同一前缀被多个请求共享时，引用增加但 block 数不变。

| 状态字段 | 物理意义 | 影响的决策 |
|---|---|---|
| `used_tokens` | 当前缓存覆盖的有效 token | 判断尾块是否已满 |
| `block_count` | 当前占用的物理 block 数 | 计算容量和新增分配量 |
| `ref_count` | 仍在使用该状态的请求数 | 大于零时禁止驱逐 |
| `hits` | 前缀被复用的次数 | 引用归零后估计保留价值 |
| `last_used` | 最近一次访问时刻 | 区分热缓存与冷缓存 |
| `free_blocks` | 总 block 减去已分配 block | 为准入和 append 提供容量约束 |

### Step 3：容量不足时如何选择动作

新增请求或 append 需要 block 时，系统先检查空闲容量，再从引用为零的缓存中选择回收对象。活动缓存不能为了接纳新请求而直接驱逐；如果安全回收后仍然不足，调度层只能排队、抢占或拒绝请求。

| 动作 | 适用条件 | 状态结果 | 主要代价 |
|---|---|---|---|
| reclaim | 引用归零且无需保留 | 立即释放 block | 失去后续复用 |
| evict | 引用归零、容量不足 | 按复用价值释放冷缓存 | 再次命中需要重算 Prefill |
| preempt + swap | 活跃请求允许暂停且有外部容量 | 活跃状态移出 GPU | 传输与恢复延迟 |
| preempt + recompute | 可接受重新执行 Prefill | 丢弃部分活动状态 | 额外计算与 TTFT |
| queue / reject | 没有安全释放路径 | 不改变活动缓存 | 排队或失败 |

![KV Cache 容量约束与安全回收顺序](../public/02_PyTorch_Algorithms/37_kv_cache_score_heap.svg)

### Step 4：实现 KV Cache 生命周期状态机

题目区用离散 block 模拟一个共享前缀缓存池。学习者需要处理 token 到 block 的取整、共享引用、跨尾块追加和释放转换；驱逐器只允许回收 `ref_count == 0` 的记录，从而把生命周期状态与容量守恒放在同一套实现中验证。

| TODO / 机制 | 学习者完成什么 | 必须保持的不变量 | 测试目标 |
|---|---|---|---|
| TODO 1 / block 取整 | 将 token 数换算为物理 block 数 | 不足一个 block 也占一个 block | 尾块边界 |
| TODO 2 / share | 命中前缀时增加引用与命中计数 | 不重复分配物理 block | 共享引用 |
| TODO 3 / append | 只为跨边界部分申请新增 block | token、block 与总容量一致 | 增量分配 |
| TODO 4 / release | 引用归零后转为 reusable 或立即 reclaim | 引用不得为负，活动记录不可驱逐 | 释放与回收 |


```python
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

```


```python
@dataclass
class CacheRecord:
    """一段可共享前缀的逻辑状态与物理 block 账本。"""

    prefix: str
    used_tokens: int
    block_count: int
    ref_count: int = 1
    hits: int = 1
    last_used: int = 0


class KVCacheLifecycleSim:
    """模拟 allocate、append、share、release、evict 与 reclaim。"""

    def __init__(self, total_blocks: int = 16, block_size_tokens: int = 4):
        if total_blocks <= 0 or block_size_tokens <= 0:
            raise ValueError('total_blocks and block_size_tokens must be positive')
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.time = 0
        self.records: Dict[str, CacheRecord] = {}
        self.log: List[str] = []

    @property
    def allocated_blocks(self) -> int:
        """返回仍被活动请求或 reusable 缓存占用的 block 数。"""
        return sum(record.block_count for record in self.records.values())

    @property
    def free_blocks(self) -> int:
        """返回可立即分配的 block 数。"""
        return self.total_blocks - self.allocated_blocks

    def _blocks_for_tokens(self, token_count: int) -> int:
        """按 block 粒度向上取整 token 数。"""
        if token_count <= 0:
            raise ValueError('token_count must be positive')
        # ==========================================
        # TODO 1（物理分配）：计算 token_count 至少需要多少个 block。
        # 注意：不能整除时也必须保留尾块；结果必须是正整数。
        # 可用变量：token_count、self.block_size_tokens
        # ==========================================
        # block_count = ???
        return block_count

    def _eviction_candidates(self) -> List[CacheRecord]:
        """只返回未被请求引用的缓存，冷且低复用的记录排在前面。"""
        candidates = [record for record in self.records.values() if record.ref_count == 0]
        return sorted(candidates, key=lambda record: (record.hits, record.last_used, -record.block_count))

    def _ensure_capacity(self, needed_blocks: int) -> None:
        """驱逐未引用缓存；不得删除仍被活动请求引用的状态。"""
        for record in self._eviction_candidates():
            if self.free_blocks >= needed_blocks:
                break
            self.records.pop(record.prefix)
            self.log.append(f'evict:{record.prefix}:{record.block_count}')
        if self.free_blocks < needed_blocks:
            raise RuntimeError('insufficient blocks: queue, preempt, or reject the request')

    def acquire(self, prefix: str, prompt_tokens: int) -> CacheRecord:
        """为新请求分配前缀，或共享已经存在的前缀状态。"""
        if not isinstance(prefix, str) or not prefix:
            raise ValueError('prefix must be a non-empty string')
        self.time += 1
        if prefix in self.records:
            record = self.records[prefix]
            # ==========================================
            # TODO 2（共享引用）：更新已有前缀的引用、命中与访问时间。
            # 注意：共享不会改变 used_tokens、block_count 或总物理占用。
            # ==========================================
            # record.ref_count = ???
            # record.hits = ???
            # record.last_used = ???
            self.log.append(f'share:{prefix}')
            return record

        block_count = self._blocks_for_tokens(prompt_tokens)
        self._ensure_capacity(block_count)
        record = CacheRecord(prefix, prompt_tokens, block_count, last_used=self.time)
        self.records[prefix] = record
        self.log.append(f'allocate:{prefix}:{block_count}')
        return record

    def append(self, prefix: str, new_tokens: int) -> int:
        """追加 Decode token，并返回本次新增的物理 block 数。"""
        if prefix not in self.records or self.records[prefix].ref_count == 0:
            raise ValueError('append requires an active cache record')
        if new_tokens <= 0:
            raise ValueError('new_tokens must be positive')
        self.time += 1
        record = self.records[prefix]
        target_tokens = record.used_tokens + new_tokens
        target_blocks = self._blocks_for_tokens(target_tokens)
        # ==========================================
        # TODO 3（尾块追加）：计算增量 block，确认容量后更新记录。
        # 注意：尾块未填满时 added_blocks 应为 0；跨边界时只申请差值。
        # ==========================================
        # added_blocks = ???
        # self._ensure_capacity(???)
        # record.used_tokens = ???
        # record.block_count = ???
        record.last_used = self.time
        self.log.append(f'append:{prefix}:{new_tokens}:{added_blocks}')
        return added_blocks

    def release(self, prefix: str, *, keep_reusable: bool = True) -> int:
        """释放一个请求引用，并返回释放后剩余的引用数。"""
        if prefix not in self.records or self.records[prefix].ref_count <= 0:
            raise ValueError('release requires an active reference')
        self.time += 1
        record = self.records[prefix]
        # ==========================================
        # TODO 4（释放与回收）：减少引用；归零且不保留时立即移除记录。
        # 注意：仍有其他引用时不得 reclaim，ref_count 不能小于 0。
        # ==========================================
        # record.ref_count = ???
        # if ???:
        #     self.records.pop(prefix)
        #     self.log.append(...)
        # else:
        #     record.last_used = self.time
        #     self.log.append(...)
        return record.ref_count

    def snapshot(self) -> List[Dict[str, int | str]]:
        """导出容量、引用和缓存状态，便于检查不变量。"""
        return [
            {
                'prefix': record.prefix,
                'state': 'active' if record.ref_count > 0 else 'reusable',
                'used_tokens': record.used_tokens,
                'block_count': record.block_count,
                'ref_count': record.ref_count,
                'hits': record.hits,
            }
            for record in sorted(self.records.values(), key=lambda item: item.prefix)
        ]

```

### 测试

运行下面的机制测试，分别定位 block 取整、共享引用、尾块追加、释放回收和容量保护问题。


```python
# 机制测试：分别验证分配粒度、共享引用、尾块追加和安全回收。
def _expect_error(error_type, action, label):
    """验证非法状态会抛出指定异常。"""
    try:
        action()
    except error_type:
        return
    raise AssertionError(f'{label} 应抛出 {error_type.__name__}')


def test_block_rounding_and_input_contract():
    """验证 token 到 block 的向上取整与输入契约。"""
    sim = KVCacheLifecycleSim(total_blocks=6, block_size_tokens=4)
    assert sim._blocks_for_tokens(1) == 1
    assert sim._blocks_for_tokens(4) == 1
    assert sim._blocks_for_tokens(5) == 2
    _expect_error(ValueError, lambda: sim._blocks_for_tokens(0), '零 token')
    _expect_error(ValueError, lambda: sim.acquire('', 4), '空 prefix')


def test_shared_reference_does_not_duplicate_blocks():
    """验证前缀共享只增加引用，不增加物理 block。"""
    sim = KVCacheLifecycleSim(total_blocks=6, block_size_tokens=4)
    first = sim.acquire('shared', 6)
    blocks_before = sim.allocated_blocks
    second = sim.acquire('shared', 6)
    assert first is second
    assert second.ref_count == 2 and second.hits == 2
    assert sim.allocated_blocks == blocks_before == 2


def test_append_allocates_only_at_tail_boundary():
    """验证 append 只在跨越尾块边界时增加 block。"""
    sim = KVCacheLifecycleSim(total_blocks=6, block_size_tokens=4)
    sim.acquire('decode', 3)
    assert sim.append('decode', 1) == 0
    assert sim.append('decode', 1) == 1
    record = sim.records['decode']
    assert (record.used_tokens, record.block_count) == (5, 2)
    assert sim.allocated_blocks == 2


def test_release_reclaim_and_active_protection():
    """验证 reusable 状态可被驱逐，而活动引用受到保护。"""
    sim = KVCacheLifecycleSim(total_blocks=5, block_size_tokens=4)
    sim.acquire('cold', 8)       # 2 blocks
    sim.release('cold')          # reusable
    sim.acquire('active', 8)     # 2 blocks, remains referenced
    sim.acquire('new', 8)        # evicts cold, then allocates 2 blocks
    assert 'cold' not in sim.records
    assert sim.records['active'].ref_count == 1
    assert sim.allocated_blocks == 4 and sim.free_blocks == 1
    sim.release('new', keep_reusable=False)
    assert 'new' not in sim.records and sim.allocated_blocks == 2
    _expect_error(ValueError, lambda: sim.append('new', 1), '已回收记录')


def test_capacity_failure_preserves_active_records():
    """验证没有安全回收对象时拒绝分配，且容量账本不被破坏。"""
    sim = KVCacheLifecycleSim(total_blocks=3, block_size_tokens=4)
    sim.acquire('a', 8)
    sim.acquire('b', 4)
    _expect_error(RuntimeError, lambda: sim.acquire('c', 4), '活动缓存占满容量')
    assert set(sim.records) == {'a', 'b'}
    assert sim.allocated_blocks == sim.total_blocks


def test_kv_cache_lifecycle():
    """汇总生命周期机制测试，并保留具体失败上下文。"""
    try:
        test_block_rounding_and_input_contract()
        test_shared_reference_does_not_duplicate_blocks()
        test_append_allocates_only_at_tail_boundary()
        test_release_reclaim_and_active_protection()
        test_capacity_failure_preserves_active_records()
        print('✅ KV Cache 生命周期测试通过：分配、共享、追加、释放和容量保护均符合预期。')
    except (NameError, UnboundLocalError) as exc:
        raise NotImplementedError('请根据变量级提示完成 TODO。') from exc


test_kv_cache_lifecycle()

```


## 参考代码与解析

### 代码


```python
# TODO：下面是题目区的参考实现。

@dataclass
class CacheRecord:
    """一段可共享前缀的逻辑状态与物理 block 账本。"""

    prefix: str
    used_tokens: int
    block_count: int
    ref_count: int = 1
    hits: int = 1
    last_used: int = 0


class KVCacheLifecycleSim:
    """模拟 allocate、append、share、release、evict 与 reclaim。"""

    def __init__(self, total_blocks: int = 16, block_size_tokens: int = 4):
        if total_blocks <= 0 or block_size_tokens <= 0:
            raise ValueError('total_blocks and block_size_tokens must be positive')
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.time = 0
        self.records: Dict[str, CacheRecord] = {}
        self.log: List[str] = []

    @property
    def allocated_blocks(self) -> int:
        """返回仍被活动请求或 reusable 缓存占用的 block 数。"""
        return sum(record.block_count for record in self.records.values())

    @property
    def free_blocks(self) -> int:
        """返回可立即分配的 block 数。"""
        return self.total_blocks - self.allocated_blocks

    def _blocks_for_tokens(self, token_count: int) -> int:
        """按 block 粒度向上取整 token 数。"""
        if token_count <= 0:
            raise ValueError('token_count must be positive')
        # ==========================================
        # TODO 1（物理分配）：计算 token_count 至少需要多少个 block。
        # 注意：不能整除时也必须保留尾块；结果必须是正整数。
        # 可用变量：token_count、self.block_size_tokens
        # ==========================================
        block_count = (token_count + self.block_size_tokens - 1) // self.block_size_tokens
        return block_count

    def _eviction_candidates(self) -> List[CacheRecord]:
        """只返回未被请求引用的缓存，冷且低复用的记录排在前面。"""
        candidates = [record for record in self.records.values() if record.ref_count == 0]
        return sorted(candidates, key=lambda record: (record.hits, record.last_used, -record.block_count))

    def _ensure_capacity(self, needed_blocks: int) -> None:
        """驱逐未引用缓存；不得删除仍被活动请求引用的状态。"""
        for record in self._eviction_candidates():
            if self.free_blocks >= needed_blocks:
                break
            self.records.pop(record.prefix)
            self.log.append(f'evict:{record.prefix}:{record.block_count}')
        if self.free_blocks < needed_blocks:
            raise RuntimeError('insufficient blocks: queue, preempt, or reject the request')

    def acquire(self, prefix: str, prompt_tokens: int) -> CacheRecord:
        """为新请求分配前缀，或共享已经存在的前缀状态。"""
        if not isinstance(prefix, str) or not prefix:
            raise ValueError('prefix must be a non-empty string')
        self.time += 1
        if prefix in self.records:
            record = self.records[prefix]
            # ==========================================
            # TODO 2（共享引用）：更新已有前缀的引用、命中与访问时间。
            # 注意：共享不会改变 used_tokens、block_count 或总物理占用。
            # ==========================================
            record.ref_count += 1
            record.hits += 1
            record.last_used = self.time
            self.log.append(f'share:{prefix}')
            return record

        block_count = self._blocks_for_tokens(prompt_tokens)
        self._ensure_capacity(block_count)
        record = CacheRecord(prefix, prompt_tokens, block_count, last_used=self.time)
        self.records[prefix] = record
        self.log.append(f'allocate:{prefix}:{block_count}')
        return record

    def append(self, prefix: str, new_tokens: int) -> int:
        """追加 Decode token，并返回本次新增的物理 block 数。"""
        if prefix not in self.records or self.records[prefix].ref_count == 0:
            raise ValueError('append requires an active cache record')
        if new_tokens <= 0:
            raise ValueError('new_tokens must be positive')
        self.time += 1
        record = self.records[prefix]
        target_tokens = record.used_tokens + new_tokens
        target_blocks = self._blocks_for_tokens(target_tokens)
        # ==========================================
        # TODO 3（尾块追加）：计算增量 block，确认容量后更新记录。
        # 注意：尾块未填满时 added_blocks 应为 0；跨边界时只申请差值。
        # ==========================================
        added_blocks = target_blocks - record.block_count
        self._ensure_capacity(added_blocks)
        record.used_tokens = target_tokens
        record.block_count = target_blocks
        record.last_used = self.time
        self.log.append(f'append:{prefix}:{new_tokens}:{added_blocks}')
        return added_blocks

    def release(self, prefix: str, *, keep_reusable: bool = True) -> int:
        """释放一个请求引用，并返回释放后剩余的引用数。"""
        if prefix not in self.records or self.records[prefix].ref_count <= 0:
            raise ValueError('release requires an active reference')
        self.time += 1
        record = self.records[prefix]
        # ==========================================
        # TODO 4（释放与回收）：减少引用；归零且不保留时立即移除记录。
        # 注意：仍有其他引用时不得 reclaim，ref_count 不能小于 0。
        # ==========================================
        record.ref_count -= 1
        if record.ref_count == 0 and not keep_reusable:
            self.records.pop(prefix)
            self.log.append(f'reclaim:{prefix}:{record.block_count}')
        else:
            record.last_used = self.time
            state = 'reusable' if record.ref_count == 0 else 'active'
            self.log.append(f'release:{prefix}:{state}:{record.ref_count}')
        return record.ref_count

    def snapshot(self) -> List[Dict[str, int | str]]:
        """导出容量、引用和缓存状态，便于检查不变量。"""
        return [
            {
                'prefix': record.prefix,
                'state': 'active' if record.ref_count > 0 else 'reusable',
                'used_tokens': record.used_tokens,
                'block_count': record.block_count,
                'ref_count': record.ref_count,
                'hits': record.hits,
            }
            for record in sorted(self.records.values(), key=lambda item: item.prefix)
        ]

```

### 解析

这组实现把“逻辑请求状态”和“物理 block 账本”分开：token 数决定需要多少 block，引用数决定能否驱逐，生命周期事件负责在两者之间维护不变量。

**TODO 1：token 到 block 的向上取整**
- `ceil(token_count / block_size)` 保证尾部不足一个 block 的 token 仍有物理空间。
- 测试覆盖 1、整除边界和跨边界三个位置，避免误写成向下取整。

**TODO 2：共享已有前缀**
- 命中只更新 `ref_count`、`hits` 和 `last_used`，不修改 `block_count`。
- 测试同时比较对象身份和总 block 数，防止“逻辑复用、物理复制”的假共享。

**TODO 3：追加 Decode token**
- 先计算追加后的目标 block 数，再只申请与当前记录之间的差值。
- 测试先填满尾块，再跨越边界，确认两次 append 分别新增 0 和 1 个 block。

**TODO 4：释放引用与回收**
- 引用归零时，`keep_reusable=True` 保留可复用缓存；关闭该选项才立即 reclaim。
- 容量驱逐只读取 `ref_count == 0` 的候选，测试验证活动缓存不会因新请求到来而被删除。

**测试结果如何解读**
- 分配或 append 失败但活动记录保持不变，说明容量失败是原子的。
- `allocated_blocks == sum(record.block_count)` 且 `free_blocks >= 0`，说明物理账本守恒。
- CPU 状态机证明机制关系；真实 backend 的命中、抢占和延迟变化需要本节 Step 5 的配对运行证据。

### Step 5：可选 GPU/backend 实验——共享前缀与容量压力

CPU 状态机验证分配、追加、共享、引用、驱逐和回收的不变量；真实 backend 实验在同一个启用 Prefix Cache 的 vLLM 服务上比较唯一前缀与共享长前缀 workload，记录 TTFT、TPOT、吞吐、失败请求和可获得的 cache/preemption 指标。

#### 5.1 环境、workload 与配对条件

本实验使用独立的 `INF-VLLM-011-CU128` 服务环境，不安装在 `INF-CUDA-MECHANISM` 中。学习者先按[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-quickstart)启动服务，并确认 Prefix Cache 已开启。

| 配置层 | baseline | candidate | 固定条件 |
|---|---|---|---|
| 前缀 | 每个请求使用不同前缀 | 多个请求共享同一长前缀 | 请求数、输出长度和正文任务一致 |
| 服务 | 同一 vLLM 实例 | 独立重启后的同配置实例 | 模型 revision、dtype、显存预算和并发一致 |
| 结果 | TTFT、TPOT、吞吐、失败和原始 `/metrics` | 同口径字段 | 交换运行先后顺序复测，避免热缓存偏差 |

#### 5.2 配置与执行

5.2 会生成两份可复查的 JSONL workload，通过统一 backend 客户端执行配对请求，并保存原始 Prometheus 指标和汇总 JSON。只修改 5.1 配置，不在执行单元中改变模型或请求口径。

```python
# 5.1 学习者入口：服务已经启动后，通常只修改 RUN_MODE 与 RUN_LABEL。
RUN_MODE = "dry_run"  # 默认值 dry_run：只登记计划；改为 real_backend 才请求真实服务。
RUN_LABEL = "default"  # 默认值 default；正式复测改为唯一标签，避免覆盖同名结果。

ENVIRONMENT_PROFILE = "INF-VLLM-011-CU128"  # 独立 Serving Profile，不与机制环境混装。
WORKLOAD_ID = "inference-kv-prefix-lifecycle"  # 稳定 workload 身份。
WORKLOAD_VERSION = "v2"  # v2 固定配对请求、原始 metrics 与汇总结果契约。
BACKEND_BASE_URL = "http://127.0.0.1:8000"  # 已启动的 OpenAI-compatible vLLM 服务地址。
MODEL_ID = "Qwen/Qwen3-0.6B"  # 服务实际加载模型；结果中还需登记 resolved revision。
NUM_PROMPTS = 12  # 两组请求数保持一致。
CONCURRENCY = 4  # 容量压力复测只改变并发或上下文长度之一。
MAX_TOKENS = 32  # 固定输出上限，避免生成长度掩盖缓存差异。
REPEATS = 1  # 单轮避免同一 prompt 因重复运行直接命中缓存。
OUTPUT_RELATIVE_PATH = f"benchmarks/results/37_kv_cache_lifecycle/{WORKLOAD_ID}_{RUN_LABEL}.json"  # 汇总结果路径。
```


```python
# 5.2 生成配对 workload、执行真实 backend 请求并保存汇总 JSON。
import json
import urllib.request
from pathlib import Path

from tools.inference_project_runtime import locate_repo_root, run_backend_benchmark

root = locate_repo_root()
output_path = root / OUTPUT_RELATIVE_PATH
result_dir = output_path.parent
workload_dir = result_dir / f"workloads_{RUN_LABEL}"

result = {
    "schema_version": "inference-mechanism-backend/v1",
    "semantic_id": "INF-KV-LIFECYCLE",
    "environment_profile": ENVIRONMENT_PROFILE,
    "run_mode": RUN_MODE,
    "run_label": RUN_LABEL,
    "workload": {
        "id": WORKLOAD_ID,
        "version": WORKLOAD_VERSION,
        "model_id": MODEL_ID,
        "num_prompts": NUM_PROMPTS,
        "concurrency": CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "repeats": REPEATS,
    },
    "baseline": None,
    "candidate": None,
    "metrics_artifacts": {},
    "failure": None,
    "evidence_level": "dry_run",
    "decision": "not_measured",
}


def read_metrics():
    """读取 backend 原始 Prometheus 文本；不可用时保留失败原因。"""
    try:
        with urllib.request.urlopen(
            BACKEND_BASE_URL.rstrip("/") + "/metrics", timeout=10
        ) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"# metrics unavailable: {exc}"


if RUN_MODE == "real_backend":
    try:
        workload_dir.mkdir(parents=True, exist_ok=True)
        shared_context = "KV Cache 生命周期包括分配、追加、共享、释放、驱逐与回收。" * 32
        unique_path = workload_dir / "unique_prefix.jsonl"
        shared_path = workload_dir / "shared_prefix.jsonl"
        unique_rows = [
            {"prompt": f"请求 {index} 的独立上下文：{index * 7919}。请概括 KV Cache。"}
            for index in range(NUM_PROMPTS)
        ]
        shared_rows = [
            {"prompt": f"{shared_context}\n请求编号 {index}：请概括上述生命周期。"}
            for index in range(NUM_PROMPTS)
        ]
        unique_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in unique_rows),
            encoding="utf-8",
        )
        shared_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in shared_rows),
            encoding="utf-8",
        )

        before_path = result_dir / f"metrics_before_{RUN_LABEL}.prom"
        middle_path = result_dir / f"metrics_after_unique_{RUN_LABEL}.prom"
        after_path = result_dir / f"metrics_after_shared_{RUN_LABEL}.prom"
        before_path.write_text(read_metrics(), encoding="utf-8")
        baseline = run_backend_benchmark(
            project="37",
            base_url=BACKEND_BASE_URL,
            model=MODEL_ID,
            label=f"unique_prefix_{RUN_LABEL}",
            output=result_dir / f"unique_prefix_{RUN_LABEL}.json",
            workload=str(unique_path.relative_to(root)),
            num_prompts=NUM_PROMPTS,
            max_tokens=MAX_TOKENS,
            concurrency=CONCURRENCY,
            warmup=1,
            repeats=REPEATS,
            backend="vllm",
            cache_policy="prefix_cache_enabled",
        )
        middle_path.write_text(read_metrics(), encoding="utf-8")
        candidate = run_backend_benchmark(
            project="37",
            base_url=BACKEND_BASE_URL,
            model=MODEL_ID,
            label=f"shared_prefix_{RUN_LABEL}",
            output=result_dir / f"shared_prefix_{RUN_LABEL}.json",
            workload=str(shared_path.relative_to(root)),
            num_prompts=NUM_PROMPTS,
            max_tokens=MAX_TOKENS,
            concurrency=CONCURRENCY,
            warmup=1,
            repeats=REPEATS,
            backend="vllm",
            cache_policy="prefix_cache_enabled",
        )
        after_path.write_text(read_metrics(), encoding="utf-8")
        result.update({
            "baseline": baseline,
            "candidate": candidate,
            "metrics_artifacts": {
                "before": str(before_path.relative_to(root)),
                "after_unique": str(middle_path.relative_to(root)),
                "after_shared": str(after_path.relative_to(root)),
            },
            "evidence_level": "backend_mechanism_benchmark",
            "decision": "tune",
        })
    except Exception as exc:
        result.update({
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "evidence_level": "failed_real_backend_attempt",
            "decision": "reject_until_environment_fixed",
        })

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### 5.3 读取配对结果

读取汇总 JSON 后先核对模型、并发、请求数、缓存策略和失败状态，再比较 TTFT、TPOT、吞吐和失败数。Prometheus 文本只记录实际 backend 暴露的指标；没有 cache hit 或 preemption 指标时应记为 unavailable，不能由延迟反推内部事件。

```python
# 5.3 只读取 5.2 保存的汇总 JSON，不重复请求 backend。
import json
from tools.inference_project_runtime import locate_repo_root

output_path = locate_repo_root() / OUTPUT_RELATIVE_PATH
if not output_path.exists():
    print(f"等待 KV 生命周期 backend 结果：{output_path}")
else:
    report = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps({
        key: report.get(key)
        for key in (
            "schema_version", "environment_profile", "workload", "failure",
            "evidence_level", "decision",
        )
    }, ensure_ascii=False, indent=2))
    for label in ("baseline", "candidate"):
        item = report.get(label) or {}
        print(label, item.get("metrics", {}))
```

#### 5.4 解释缓存复用与容量证据

先检查 `failure` 和请求口径，再比较唯一前缀与共享前缀。TTFT 改善只能说明当前固定 workload 下存在可复测信号；只有 backend 指标或 trace 明确记录命中、抢占和缓存占用时，才能把变化归因到 KV 生命周期事件。

| 观察项 | 读取字段或资产 | 判断问题 |
|---|---|---|
| 请求口径 | `workload`、两份 JSONL | baseline/candidate 是否只改变前缀共享关系 |
| 服务性能 | `metrics` 中的 TTFT、TPOT、吞吐、失败 | 共享前缀是否改善首 Token，同时没有恶化生成阶段与稳定性 |
| 生命周期 | 三份 Prometheus 快照 | 是否能直接观察 cache usage、命中或 preemption；缺失时写 unavailable |
| 顺序偏差 | 独立重启并交换运行顺序的复测 | 结果是否只是热缓存或先后顺序造成 |
| 证据与决策 | `failure`、`evidence_level`、`decision` | 当前只能继续调参，还是已具备项目复测条件 |
## 相关阅读

完成生命周期状态机后，可继续观察真实 backend 的前缀复用、服务准入和调度结果。

- [PagedAttention 原论文：Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- [vLLM 官方 Prefix Caching 文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [38. Prefill/Decode Scheduling | Prefill/Decode 调度](./38_Prefill_Decode_Scheduling.md)
- [69. Prefix Caching Benchmark | 前缀缓存基准项目](./69_Prefix_Caching_Benchmark.md)
- [70. Serving Scheduler Benchmark | 服务调度基准项目](./70_Serving_Scheduler_Benchmark.md)
