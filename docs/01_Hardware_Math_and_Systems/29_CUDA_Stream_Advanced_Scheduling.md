# 29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度

**难度：** Hard | **环境：** GPU optional | **标签：** `算子编程`, `CUDA Stream`, `异步调度` | **目标人群：** 编译与系统入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

基础 stream 概念回答的是“能不能并起来”，而高级调度回答的是“怎么并才不乱”。只要事件依赖、优先级和任务回放没有排好，多 stream 不但不能提升吞吐，反而会把等待链变得更难分析。

本节沿着“依赖关系 → event 同步 → stream 优先级 → graph 回放”组织执行时间线。完成后，你应能区分可并行工作与必须等待的依赖，并判断通信或搬运中有多少时间真正暴露在关键路径上。

**关键词：** `stream`, `event`, `graph`

---
## 前置阅读

**导语：** 先把 CPU/GPU 异构调度、CUDA stream 基础和 block 组织接上，再看这一页的高级调度，会更容易把“怎么排任务”讲清楚。

- [07. CPU and GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度](./07_CPU_GPU_Heterogeneous_Scheduling.md)
- [17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](./17_CUDA_Stream_and_Asynchrony.md)
- [18. Triton Block Model | Triton Block 模型](./18_Triton_Block_Model.md)

## 相关阅读

**导语：** 将 stream 与 event 的依赖判断继续接到真实 collective 和通信重叠实验。

- [Part 02 · 47 NCCL 通信剖析](../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md)
- [Part 02 · 48 通信热点与缓解](../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
---
## Q1：CUDA Stream 为什么能做并行调度？

<details>
<summary>点击展开查看解析</summary>

CUDA Stream 可以看成一条命令队列：
- 同一条 Stream 内部的操作顺序执行
- 不同 Stream 之间在硬件允许时可以并行

这允许我们把任务拆成：
- 数据搬运
- 计算
- 后处理

只要这些阶段之间依赖关系允许，就可以把它们放到不同 Stream 里，从而提高设备利用率。

一个数量级直觉表可以帮助你判断“为什么要调度”：

| 操作 | 典型耗时 | 说明 |
| --- | --- | --- |
| 空 kernel launch | `5-10 μs` | CPU 下发指令的开销 |
| 小算子计算 | `1-2 μs` | GPU 实际计算很短 |
| 单 kernel 总耗时 | `6-12 μs` | launch 往往占主导 |
| CUDA Graph 回放 | `<1 μs` | 几乎不再重复 launch |

所以，Stream / Graph 的价值并不是“让 GPU 变魔法一样更强”，而是尽量少浪费在调度和启动上。
</details>

### Q1小验证：Stream 流水线与重叠收益


```python
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class TaskStage:
    name: str
    duration_us: float
    stream: str
    depends_on: Tuple[str, ...] = ()

def sequential_time_us(stages: List[TaskStage]) -> float:
    """假设所有阶段串行执行的总时间。"""
    return sum(stage.duration_us for stage in stages)

def pipelined_time_us(stages: List[TaskStage]) -> float:
    """非常粗略地估算流水线调度后的总时间。"""
    stream_totals: Dict[str, float] = {}
    for stage in stages:
        stream_totals.setdefault(stage.stream, 0.0)
        stream_totals[stage.stream] += stage.duration_us
    return max(stream_totals.values()) if stream_totals else 0.0

def overlap_ratio(sequential_us: float, pipelined_us: float) -> float:
    if sequential_us == 0:
        return 0.0
    return 1 - pipelined_us / sequential_us

```

## Q2：CUDA Event 为什么重要？

<details>
<summary>点击展开查看解析</summary>

Event 的作用是跨 Stream 做同步点。

你可以把它理解成：
- 某个 Stream 先完成一段工作
- 记录一个 Event
- 另一个 Stream 在等待这个 Event 后继续执行

这比“全局阻塞”更精细，因为它只同步真正有依赖关系的部分。

所以在复杂流水线里，Event 往往是把异步调度真正串起来的关键。

Event 的开销通常很小，但它不是“零成本”；工程上常见的用法是只在真正有依赖关系的地方插入 Event，避免把同步点铺得太密。
</details>

### Q2小验证：Event 依赖关系


```python
def build_dependency_edges(stages: List[TaskStage]) -> List[Tuple[str, str]]:
    edges = []
    for stage in stages:
        for dep in stage.depends_on:
            edges.append((dep, stage.name))
    return edges

def has_cross_stream_dependency(stages: List[TaskStage]) -> bool:
    name_to_stage = {s.name: s for s in stages}
    for stage in stages:
        for dep in stage.depends_on:
            if name_to_stage[dep].stream != stage.stream:
                return True
    return False

```


```python
def test_event_dependencies():
    stages = [
        TaskStage('H2D', 12, 'copy'),
        TaskStage('Kernel', 40, 'compute', depends_on=('H2D',)),
        TaskStage('D2H', 10, 'copy', depends_on=('Kernel',)),
        TaskStage('Post', 8, 'post', depends_on=('D2H',)),
    ]

    edges = build_dependency_edges(stages)
    assert ('H2D', 'Kernel') in edges
    assert ('Kernel', 'D2H') in edges
    assert ('D2H', 'Post') in edges
    assert has_cross_stream_dependency(stages) is True
    print('✅ Event 依赖测试通过')

test_event_dependencies()

```

## Q3：CUDA Graph 和 Stream 调度是什么关系？

<details>
<summary>点击展开查看解析</summary>

CUDA Graph 更像是把一整段稳定的执行路径捕获下来，再在后续回放。

它的价值在于：
- 降低频繁 kernel launch 的开销
- 减少 Python 或调度层的干预
- 在推理场景中稳定化执行路径

但它也有局限：
- 输入 shape 如果经常变化，捕获和回放就不稳定
- 依赖关系复杂时，图捕获也更难管理

所以 Graph 常和 Stream 一起出现，但它解决的是“固定流程回放”，不是替代所有异步调度。

什么时候值得用 Graph？
- 固定 batch size + 固定 seq length 的离线批处理推理
- 多次重复执行相同的计算模式，例如固定层数的 Transformer

什么时候不太划算？
- 变长请求很频繁的在线推理
- 动态 shape 经常变化、每次路径都不同的任务

可以把它记成一句话：Graph 适合“路径稳定”，不适合“形状经常跳”。
</details>

### Q3小验证：Graph 适用性判断


```python
def graph_suitability(is_fixed_shape: bool, is_repeated_path: bool, has_many_branches: bool) -> bool:
    """非常简化的 CUDA Graph 适用性判断。"""
    return is_fixed_shape and is_repeated_path and not has_many_branches

assert graph_suitability(True, True, False) is True
assert graph_suitability(True, False, False) is False
assert graph_suitability(False, True, False) is False
assert graph_suitability(True, True, True) is False
print('✅ Graph 适用性测试通过')

```


```python
cases = [
    ('离线批处理推理', True, True, False),
    ('变长在线推理', False, True, False),
    ('动态分支很多的控制流', True, True, True),
]

for name, fixed_shape, repeated_path, many_branches in cases:
    print(f'{name:<18s}:', '适合 Graph' if graph_suitability(fixed_shape, repeated_path, many_branches) else '不太适合 Graph')

```

## Q4：Stream 优先级和典型流水线怎么理解？

<details>
<summary>点击展开查看解析</summary>

CUDA 支持高 / 低优先级 Stream。高优先级 Stream 中的 kernel 会更容易被优先调度，适合放高优请求或更敏感的控制流。

一个常见的三段流水线思路是：

```text
Stream A: H2D
Stream B: Kernel
Stream C: D2H + 后处理
```

更实用的做法不是把所有任务都丢进同一条 Stream，而是把职责拆开：
- 一个 Stream 负责输入搬运
- 一个 Stream 负责核心计算
- 一个 Stream 负责输出回传和收尾

再用 Event 连接依赖关系，这样更容易让搬运和计算重叠。
</details>

### Q4小验证：优先级与流水线


```python
def should_split_stream(copy_us: float, compute_us: float, post_us: float) -> bool:
    # 只有当搬运 / 计算 / 收尾三段都有可分离职责时，拆 Stream 才更有意义。
    total = copy_us + compute_us + post_us
    if total <= 0:
        return False
    copy_ratio = copy_us / total
    compute_ratio = compute_us / total
    return copy_ratio >= 0.1 and compute_ratio >= 0.3


def recommended_pipeline(copy_us: float, compute_us: float, post_us: float) -> Dict[str, str]:
    if not should_split_stream(copy_us, compute_us, post_us):
        return {'decision': 'single_stream', 'reason': '重叠收益有限'}
    return {
        'decision': 'split_streams',
        'reason': '搬运 / 计算 / 收尾职责可分离，适合用 Event 串依赖',
    }

print(recommended_pipeline(12, 40, 10))
print(recommended_pipeline(2, 60, 1))

```

## Q5：高级调度最常见的误区是什么？

<details>
<summary>点击展开查看解析</summary>

- **“Stream 越多越好”**  
  不对。太多 Stream 会增加调度复杂度。

- **“异步一定更快”**  
  不对。只有当传输和计算能够重叠时，异步才更有意义。

- **“Event 就是锁”**  
  不准确。Event 是同步点，不是全局锁。

- **“CUDA Graph 适合所有推理”**  
  不对。动态 shape 很多时，Graph 的收益会下降。
</details>
