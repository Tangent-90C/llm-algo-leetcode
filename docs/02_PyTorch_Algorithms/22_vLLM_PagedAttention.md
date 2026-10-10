# 22. vLLM PagedAttention | vLLM 分页注意力
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `KV Cache`, `PagedAttention` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线推理服务里，请求不会整齐地一起开始、一起结束：有的 prompt 很短，有的上下文很长，有的很快生成完，有的还在继续 decode。如果仍然按最大长度给每个请求预留整块 KV Cache，显存会被大量浪费，GPU 也会因为静态 batch 里的空洞而等待。

PagedAttention 的思路是把 KV Cache 像分页内存一样管理：物理显存切成固定大小的 block，每个请求只维护一张逻辑到物理的 block table。prefill 时按需申请，decode 跨块时再补新 block，真正计算时再按表拼回逻辑连续的缓存。本节会用一个简化管理器模拟这条链路，看清 vLLM 为什么能支撑高吞吐在线 serving。

**关键词：** `PagedAttention`, `KV cache`, `block table`

---

## 前置阅读

**导语：** 先理解单步解码如何读取 KV Cache，再把连续缓存和分页 block 的管理方式放到同一条请求链路中比较。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)

---

### Step 1: PagedAttention 如何组织逻辑与物理 KV Cache
在线请求的长度和结束时间并不相同，如果每个请求都按最大长度预留连续 KV Cache，就会同时产生未使用尾部和难以复用的空闲空间。PagedAttention 的输入是请求 token 与缓存预算，处理方式是按需分配固定大小的物理 Block，并用 Block Table 记录逻辑位置到物理 Block 的映射；输出是可继续增长、可回收的缓存布局。
先通过表格建立连续分配、固定 Block 和分页映射的整体关系，再观察主图中的逻辑访问路径。

| 管理方式 | 如何分配 KV Cache | 主要问题或收益 |
|---|---|---|
| 连续预留 | 按 `max_len` 一次分配连续空间 | 长度未知时产生尾部浪费和空洞 |
| 虚拟内存类比 | 用逻辑位置访问物理存储 | 逻辑地址与物理地址可以分离 |
| 固定 Block | 每个 Block 容纳固定数量 token，例如 16 个 | 分配粒度统一，便于回收和复用 |
| 分页管理 | 按需分配固定大小的物理 Block | 物理块可以不连续，空间更容易复用 |
| Block Table | 记录逻辑 Block 到物理 Block 的映射 | 计算时仍能按逻辑 token 顺序访问缓存 |

![PagedAttention 的 KV Cache 管理总览](../public/02_PyTorch_Algorithms/22_paged_attention_overview.svg)

### Step 2: 物理 Block 如何分配、扩容与释放
`BlockTable` 的状态会随着请求推进而变化：prefill 建立初始映射，decode 只在跨过边界时扩容，请求完成后释放物理块。沿着这个生命周期观察“请求长度变化”如何转化为“物理块状态变化”。

| 阶段 | 输入 | 状态变化 | 需要检查的边界 |
|---|---|---|---|
| Prefill | prompt 长度 | 申请多个物理 Block | 空闲 Block 是否足够 |
| Decode | 新增 token | 跨边界时追加 Block | 扩容失败时状态是否回滚 |
| Release | 已完成请求 | 归还 Block 并清空块表 | 是否重复释放 |
| Reuse | 新请求 | 使用已释放的物理 Block | 物理 Block 是否允许不连续 |

### Step 3: Block Table 如何把逻辑访问映射到物理缓存

分页缓存需要把逻辑 token 顺序与非连续物理空间对应起来。物理池保存 Block，Block Table 保存逻辑到物理的映射，管理器负责分配、释放和按需扩容；下面的表格用不变量说明三者如何协作。

| 数据结构 | 作用 | 应保持的不变量 |
|---|---|---|
| `physical_pool` | 保存物理 KV Block | 一个物理索引对应一个池内 Block |
| `block_table` | 逻辑 Block 到物理 Block 的映射 | 不应出现无效或重复的物理索引 |
| `free_blocks` | 管理尚未分配的物理 Block | 已分配 Block 不应同时出现在其中 |
| `seq_len` | 判断尾块和 decode 扩容边界 | 与当前逻辑 token 数一致 |

![PagedAttention 块表图](../public/02_PyTorch_Algorithms/22_paged_attention_blocks.svg)

### Step 4: 实现并验证分页缓存管理器

本题用固定大小的张量池和请求块表复现分页缓存；请求状态、占用报告、释放校验和非核心控制流已经提供，学习者补全容量账本、Prefill 原子分配、Decode 跨块扩容和逻辑缓存恢复。测试覆盖正常增长、非连续映射、OOM 回滚与释放复用。

| 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| `estimate_kv_cache_bytes` | TODO 1：计算单 token、单 Block 和总容量 | 同时计入 K/V、层数、KV Head 和数据类型 | 容量账本测试 |
| `allocate_for_prefill` | TODO 2：向上取整并选择待提交物理块 | OOM 与重复分配不改变请求和空闲池 | Prefill 与原子 OOM 测试 |
| `allocate_for_decode` | TODO 3：识别追加 token 是否跨越 Block 边界 | 仅跨界时申请；失败时长度和块表不变 | Decode 边界与回滚测试 |
| `get_physical_cache` | TODO 4：按 Block Table 恢复逻辑顺序 | 不按物理索引排序，并截断尾块 | 非连续物理块恢复测试 |



```python
import torch
from typing import Dict, List, Tuple
```


```python
def estimate_kv_cache_bytes(num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """估算 KV Cache 理论容量；不代表 vLLM allocator 的实际峰值。"""
    values = (num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('所有容量维度和 dtype_bytes 都必须为正数')
    # TODO 1（容量账本）：依次计算单 token、单 Block 和整个池的理论容量。
    # kv_bytes_per_token = ???
    # kv_bytes_per_block = ???
    # total_cache_bytes = ???
    raise NotImplementedError('TODO 1：请完成 KV Cache 容量账本')
    layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]
    return {'kv_bytes_per_token': kv_bytes_per_token, 'kv_bytes_per_block': kv_bytes_per_block, 'total_cache_bytes': total_cache_bytes, 'layout_shape': layout_shape}


class Request:
    """记录逻辑序列长度，以及逻辑 Block 到物理 Block 的映射。

    `block_table[i]` 是第 i 个逻辑 Block 对应的物理索引；物理索引不要求连续。"""

    def __init__(self, request_id: int, prompt_len: int):
        if request_id < 0 or prompt_len <= 0:
            raise ValueError('request_id 必须非负，prompt_len 必须为正数')
        self.request_id = request_id
        self.seq_len = prompt_len
        # 记录此请求占据的物理 Block 索引
        self.block_table: List[int] = []

class KVCacheManager:
    """用空闲块列表模拟 KV Cache 的分配、扩容、释放和恢复。

    这里的物理池只保存教学用张量，不等同于 vLLM 的真实 KV Cache layout。"""
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        if num_blocks <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError('num_blocks、block_size 和 head_dim 必须为正数')
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # 模拟预分配一块大显存池 (vLLM 会在 GPU 上分配几 GB)
        # 形状: [num_blocks, block_size, head_dim]
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        
        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))
    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        if req.block_table:
            raise ValueError('请求已经完成 Prefill，不能重复分配')
        # TODO 2（Prefill 分配）：先计算目标块数，再选择待提交的物理块。
        # needed_blocks = ???
        # allocated_blocks = ???
        raise NotImplementedError('TODO 2：请完成 Prefill 原子分配')
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError('OOM')
        del self.free_blocks[:needed_blocks]
        req.block_table.extend(allocated_blocks)

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        # TODO 3（Decode 扩容）：计算追加后的长度，并判断是否进入新 Block。
        # new_seq_len = ???
        # is_new_block_needed = ???
        raise NotImplementedError('TODO 3：请判断 Decode 是否跨越 Block 边界')
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError('OOM')
            req.block_table.append(self.free_blocks.pop(0))
        req.seq_len = new_seq_len

    def allocation_report(self, req: Request) -> Dict[str, float]:
        """报告逻辑 token、已分配 token、尾块浪费和利用率。"""
        allocated_tokens = len(req.block_table) * self.block_size
        unused_tail_tokens = allocated_tokens - req.seq_len
        utilization = req.seq_len / allocated_tokens if allocated_tokens else 0.0
        return {'logical_tokens': req.seq_len, 'allocated_tokens': allocated_tokens, 'unused_tail_tokens': unused_tail_tokens, 'utilization': utilization}

    def release_request(self, req: Request):
        """释放请求占用的物理块，并清空其块表。"""
        released_block_ids = list(req.block_table)
        if not released_block_ids:
            raise ValueError('请求没有可释放的物理块，可能已经释放')
        if len(set(released_block_ids)) != len(released_block_ids):
            raise ValueError('block_table 不能包含重复物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in released_block_ids):
            raise ValueError('请求包含越界物理块')
        if any(block_id in self.free_blocks for block_id in released_block_ids):
            raise ValueError('请求包含已释放的物理块')
        self.free_blocks.extend(released_block_ids)
        self.free_blocks.sort()
        req.block_table.clear()

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        if not req.block_table:
            raise ValueError('请求没有可读取的物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in req.block_table):
            raise ValueError('block_table 包含越界物理块')
        # TODO 4（逻辑恢复）：按 Block Table 顺序读取并沿 token 维拼接。
        # blocks = ???
        # cat_blocks = ???
        raise NotImplementedError('TODO 4：请恢复逻辑连续的 KV Cache')
        return cat_blocks[:req.seq_len]


```

### 测试

运行下面的测试单元，确认 prefill / decode / cache 拼装三段链路都正确。

```python
# 机制测试：按输入契约、生命周期、逻辑恢复、原子 OOM 和释放复用分别定位问题。
def _legacy_paged_attention_manager():
    try:
        # Case 1: 典型 Prefill + Decode + Cache 拼装
        manager = KVCacheManager(num_blocks=10, block_size=4, head_dim=64)
        print("初始化内存池...")

        for bad_request in ((-1, 2), (1, 0)):
            try:
                Request(request_id=bad_request[0], prompt_len=bad_request[1])
                raise AssertionError('非法请求参数未被拒绝')
            except ValueError:
                pass
        print("✅ Request 输入校验通过！")

        req1 = Request(request_id=1, prompt_len=6)
        manager.allocate_for_prefill(req1)
        assert len(req1.block_table) == 2, "长度 6 的请求应分配 2 个 Block！"
        assert len(manager.free_blocks) == 8, "池中应该剩下 8 个空闲块！"
        print(f"✅ Prefill 测试通过！Req1 分配的块表: {req1.block_table}")

        try:
            manager.allocate_for_prefill(req1)
        except ValueError:
            print("✅ 重复 Prefill 已拒绝！")
        else:
            raise AssertionError('同一请求不能重复 Prefill 并追加 Block')

        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 2, "生成第 7 个 token 时不应该分配新块！"

        manager.allocate_for_decode(req1)
        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 3, "生成第 9 个 token 时应当分配了第 3 个新块！"
        assert len(manager.free_blocks) == 7, "池中应该剩下 7 个空闲块！"
        print(f"✅ Decode 动态分配测试通过！Req1 最新块表: {req1.block_table}")

        for block_id, value in zip(req1.block_table, [1.0, 2.0, 3.0]):
            manager.physical_kv_cache[block_id].fill_(value)
        cache = manager.get_physical_cache(req1)
        assert cache.shape == (9, 64), f"拼装出来的连续 Cache 形状不对，应为 (9, 64)，实为 {cache.shape}"
        assert torch.all(cache[:4] == 1.0), "第 1 个 Block 未正确拼装！"
        assert torch.all(cache[4:8] == 2.0), "第 2 个 Block 未正确拼装！"
        assert torch.all(cache[8:] == 3.0), "第 3 个 Block 的截断拼装不正确！"
        print("✅ Cache 拼装测试通过！多块物理缓存被正确恢复为逻辑连续序列。")

        # Case 1b: 物理块可以不连续，但恢复必须遵循逻辑 Block 顺序
        scattered_manager = KVCacheManager(num_blocks=4, block_size=2, head_dim=2)
        scattered_req = Request(request_id=10, prompt_len=3)
        scattered_req.block_table = [3, 1]
        scattered_manager.physical_kv_cache[3].fill_(3.0)
        scattered_manager.physical_kv_cache[1].fill_(1.0)
        scattered_cache = scattered_manager.get_physical_cache(scattered_req)
        assert torch.all(scattered_cache[:2] == 3.0)
        assert torch.all(scattered_cache[2:] == 1.0)
        print("✅ 非连续物理块按逻辑顺序恢复通过！")

        # Case 2: 恰好跨越 block 边界时，Decode 应该分配新块，并正确截断最后一块
        manager2 = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
        req2 = Request(request_id=2, prompt_len=4)
        manager2.allocate_for_prefill(req2)
        assert len(req2.block_table) == 1, "长度 4 的请求应只分配 1 个 Block！"
        manager2.allocate_for_decode(req2)
        assert len(req2.block_table) == 2, "长度 5 的请求应分配第 2 个 Block！"
        manager2.physical_kv_cache[req2.block_table[0]].fill_(7.0)
        manager2.physical_kv_cache[req2.block_table[1]].fill_(8.0)
        cache2 = manager2.get_physical_cache(req2)
        assert cache2.shape == (5, 8), f"拼装出来的连续 Cache 形状不对，应为 (5, 8)，实为 {cache2.shape}"
        assert torch.all(cache2[:4] == 7.0), "边界块的前 4 个 token 不正确！"
        assert torch.all(cache2[4:] == 8.0), "边界块的最后 1 个 token 不正确！"
        print("✅ 边界分配与截断测试通过！")

        # Case 3: OOM 分支必须抛出 RuntimeError
        oom_manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
        oom_req = Request(request_id=3, prompt_len=5)
        try:
            oom_manager.allocate_for_prefill(oom_req)
        except RuntimeError as e:
            assert "OOM" in str(e), "OOM 异常信息不正确！"
            print("✅ OOM 测试通过！")
        else:
            raise AssertionError('显存池不足时应该抛出 RuntimeError("OOM")！')
        # Case 4: Prefill OOM 不应留下半分配状态
        atomic_manager = KVCacheManager(num_blocks=2, block_size=4, head_dim=8)
        atomic_req = Request(request_id=4, prompt_len=9)
        free_before = list(atomic_manager.free_blocks)
        try:
            atomic_manager.allocate_for_prefill(atomic_req)
        except RuntimeError as e:
            assert 'OOM' in str(e), '应明确报告 OOM'
            assert atomic_req.block_table == [], 'OOM 后请求不应保留半分配块'
            assert atomic_manager.free_blocks == free_before, 'OOM 后空闲池不应被部分消耗'
            print('✅ Prefill 原子 OOM 测试通过！')
        else:
            raise AssertionError('资源不足时必须抛出 OOM')

        ledger = estimate_kv_cache_bytes(num_blocks=4, block_size=4, num_layers=2, num_kv_heads=8, head_dim=16, dtype_bytes=2)
        assert ledger['kv_bytes_per_token'] == 1024, 'K/V 两份缓存账本计算不正确'
        assert ledger['total_cache_bytes'] == 16384, 'Block 池总容量计算不正确'
        reusable_manager = KVCacheManager(num_blocks=3, block_size=4, head_dim=8)
        reusable_req = Request(request_id=5, prompt_len=5)
        reusable_manager.allocate_for_prefill(reusable_req)
        allocated_before_release = list(reusable_req.block_table)
        report = reusable_manager.allocation_report(reusable_req)
        assert report['allocated_tokens'] == 8, '已分配 token 账本不正确'
        assert report['unused_tail_tokens'] == 3, '尾块浪费计算不正确'
        assert round(report['utilization'], 3) == 0.625, 'Block 利用率计算不正确'
        reusable_manager.release_request(reusable_req)
        assert reusable_req.block_table == [], '释放后请求块表应为空'
        assert reusable_manager.free_blocks == [0, 1, 2], '释放后的 Block 应可复用'
        try:
            reusable_manager.release_request(reusable_req)
        except ValueError:
            pass
        else:
            raise AssertionError('重复释放请求应明确报错')
        rollback_manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
        rollback_req = Request(request_id=6, prompt_len=4)
        rollback_manager.allocate_for_prefill(rollback_req)
        old_seq_len = rollback_req.seq_len
        try:
            rollback_manager.allocate_for_decode(rollback_req)
        except RuntimeError:
            assert rollback_req.seq_len == old_seq_len, 'Decode OOM 后 seq_len 不应改变'
            print('✅ Decode OOM 回滚测试通过！')
        else:
            raise AssertionError('Decode 跨块时应触发 OOM')
        print('✅ KV Cache 账本与释放复用测试通过！')

        print("\n✅ All Tests Passed! PagedAttention 内存管理逻辑验证通过。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except Exception as e:
        # 只把 TODO 未完成单独提示；断言、类型和状态错误保留原始异常，便于定位。
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        raise


def test_request_and_ledger_contract():
    """验证请求输入契约与 KV Cache 容量账本。"""
    for bad_request in ((-1, 2), (1, 0)):
        try:
            Request(request_id=bad_request[0], prompt_len=bad_request[1])
        except ValueError:
            continue
        raise AssertionError('非法请求参数应被拒绝')
    ledger = estimate_kv_cache_bytes(4, 4, 2, 8, 16, 2)
    assert ledger['kv_bytes_per_token'] == 1024
    assert ledger['total_cache_bytes'] == 16384

def test_prefill_and_decode_growth():
    """验证 Prefill 分配和 Decode 跨 Block 边界扩容。"""
    manager = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
    req = Request(request_id=1, prompt_len=4)
    manager.allocate_for_prefill(req)
    assert len(req.block_table) == 1
    manager.allocate_for_decode(req)
    assert req.seq_len == 5 and len(req.block_table) == 2

def test_logical_cache_reconstruction():
    """验证非连续物理块按逻辑 Block 顺序恢复。"""
    manager = KVCacheManager(num_blocks=4, block_size=2, head_dim=2)
    req = Request(request_id=2, prompt_len=3)
    req.block_table = [3, 1]
    manager.physical_kv_cache[3].fill_(3.0)
    manager.physical_kv_cache[1].fill_(1.0)
    cache = manager.get_physical_cache(req)
    assert cache.shape == (3, 2)
    assert torch.all(cache[:2] == 3.0) and torch.all(cache[2:] == 1.0)

def test_oom_is_atomic():
    """验证 Prefill OOM 不会留下半分配状态。"""
    manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
    req = Request(request_id=3, prompt_len=5)
    free_before = list(manager.free_blocks)
    try:
        manager.allocate_for_prefill(req)
    except RuntimeError as error:
        assert 'OOM' in str(error)
    else:
        raise AssertionError('资源不足时应抛出 OOM')
    assert req.block_table == [] and manager.free_blocks == free_before

def test_release_and_reuse():
    """验证尾块账本、释放和物理 Block 复用。"""
    manager = KVCacheManager(num_blocks=3, block_size=4, head_dim=8)
    req = Request(request_id=4, prompt_len=5)
    manager.allocate_for_prefill(req)
    report = manager.allocation_report(req)
    assert report['unused_tail_tokens'] == 3
    manager.release_request(req)
    assert req.block_table == [] and manager.free_blocks == [0, 1, 2]

def test_paged_attention_manager():
    """汇总运行分页缓存的容量、扩容、映射、OOM 与释放测试。"""
    test_request_and_ledger_contract()
    test_prefill_and_decode_growth()
    test_logical_cache_reconstruction()
    test_oom_is_atomic()
    test_release_and_reuse()
    print('✅ PagedAttention 机制测试通过：账本、扩容、映射、OOM 和回收均通过。')


test_paged_attention_manager()

```


### 参考答案

```python
class Request:
    """记录逻辑序列长度，以及逻辑 Block 到物理 Block 的映射。

    `block_table[i]` 是第 i 个逻辑 Block 对应的物理索引；物理索引不要求连续。"""

    def __init__(self, request_id: int, prompt_len: int):
        if request_id < 0 or prompt_len <= 0:
            raise ValueError('request_id 必须非负，prompt_len 必须为正数')
        self.request_id = request_id
        self.seq_len = prompt_len
        self.block_table: List[int] = []

def estimate_kv_cache_bytes(num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """估算 KV Cache 理论容量；不代表 vLLM allocator 的实际峰值。"""
    values = (num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('所有容量维度和 dtype_bytes 都必须为正数')
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    kv_bytes_per_block = block_size * kv_bytes_per_token
    total_cache_bytes = num_blocks * kv_bytes_per_block
    layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]
    return {'kv_bytes_per_token': kv_bytes_per_token, 'kv_bytes_per_block': kv_bytes_per_block, 'total_cache_bytes': total_cache_bytes, 'layout_shape': layout_shape}


class KVCacheManager:
    """用空闲块列表模拟 KV Cache 的分配、扩容、释放和恢复。

    这里的物理池只保存教学用张量，不等同于 vLLM 的真实 KV Cache layout。"""
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        if num_blocks <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError('num_blocks、block_size 和 head_dim 必须为正数')
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # 给定实现：模拟预分配一块大显存池
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        
        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))
    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        if req.block_table:
            raise ValueError('请求已经完成 Prefill，不能重复分配')
        # TODO 2（Prefill 分配）：先计算目标块数，再选择待提交的物理块。
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError("OOM")
        
        allocated_blocks = self.free_blocks[:needed_blocks]
        del self.free_blocks[:needed_blocks]
        req.block_table.extend(allocated_blocks)

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        new_seq_len = req.seq_len + 1
        
        # TODO 3（Decode 扩容）：计算追加后的长度，并判断是否进入新 Block。
        is_new_block_needed = (new_seq_len % self.block_size) == 1
        
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError("OOM")
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)
        req.seq_len = new_seq_len

    def allocation_report(self, req: Request) -> Dict[str, float]:
        allocated_tokens = len(req.block_table) * self.block_size
        unused_tail_tokens = allocated_tokens - req.seq_len
        utilization = req.seq_len / allocated_tokens if allocated_tokens else 0.0
        return {'logical_tokens': req.seq_len, 'allocated_tokens': allocated_tokens, 'unused_tail_tokens': unused_tail_tokens, 'utilization': utilization}

    def release_request(self, req: Request):
        released_block_ids = list(req.block_table)
        if not released_block_ids:
            raise ValueError('请求没有可释放的物理块，可能已经释放')
        if len(set(released_block_ids)) != len(released_block_ids):
            raise ValueError('block_table 不能包含重复物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in released_block_ids):
            raise ValueError('请求包含越界物理块')
        if any(block_id in self.free_blocks for block_id in released_block_ids):
            raise ValueError('请求包含已释放的物理块')
        self.free_blocks.extend(released_block_ids)
        self.free_blocks.sort()
        req.block_table.clear()

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        # TODO 4（逻辑恢复）：按 Block Table 顺序读取并沿 token 维拼接。
        if not req.block_table:
            raise ValueError('请求没有可读取的物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in req.block_table):
            raise ValueError('block_table 包含越界物理块')
        blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]
        cat_blocks = torch.cat(blocks, dim=0)
        
        # 只截取真实 seq_len 长度返回
        return cat_blocks[:req.seq_len]


```

### 解析

题目把报告和释放校验作为给定骨架，解析只对应四个需要补全的分页机制。

**TODO 1：KV Cache 容量账本**
单 token 容量同时包含 K 和 V，并乘以层数、KV Head 数、Head 维度与元素字节数；再按 Block 大小和 Block 数逐级汇总。容量测试检查每一级结果。

**TODO 2：Prefill 原子分配**
请求长度按 Block 大小向上取整。候选物理块先从空闲池中选出，只有资源充足时才同时提交空闲池和块表，因此 OOM 不会留下半分配状态。

**TODO 3：Decode 跨块扩容**
追加 token 后只有进入新 Block 的第一个位置才需要申请物理块。测试同时检查普通增长、边界增长和无空闲块时的状态回滚。

**TODO 4：恢复逻辑缓存**
物理索引可以不连续，读取顺序必须来自 `block_table`；拼接后按 `seq_len` 截断尾块。非连续映射测试用于防止错误地按物理索引排序。

### 真实 GPU 验证出口

本节不使用合成张量模拟 PagedAttention 的 GPU 收益。Step 4 已验证 block 分配、逻辑映射、尾块碎片与原子 OOM；真实 GPU 证据需要启动支持 PagedAttention 的 Serving backend，并在固定模型、请求长度、block size 和并发度后记录 KV 容量、失败请求、TTFT 与 TPOT。

需要观察缓存引用、驱逐与回收时，继续到 [37 KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)；需要端到端前缀复用证据时，使用 [69 前缀缓存基准](./69_Prefix_Caching_Benchmark.md)。
## 相关阅读

**论文与官方实现**

完成本节的 Block Table 和物理块管理模拟后，可以沿着“分页缓存机制 → 推理引擎实现 → 缓存复用与调度”继续阅读。

- [Efficient Memory Management for Large Language Model Serving with PagedAttention 原论文](https://arxiv.org/abs/2309.06180)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [24. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./24_Prefix_Cache_Matching_and_Reuse.md)
- [34. SGLang RadixAttention | SGLang 基数注意力](./34_SGLang_RadixAttention.md)
- [37. KV Cache Lifecycle and Capacity Control | KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)
