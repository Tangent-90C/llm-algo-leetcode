# 23. Speculative Decoding | 投机解码
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `解码`, `Speculative Decoding` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

自回归生成中，目标模型通常要逐个 token 推进；当每一步都要重复进入模型时，单步成本会直接累积到整段输出。本节从已有的 KV Cache 状态视角出发，引入草稿模型：先提出多个候选，再由目标模型集中验证。

学习路径沿着“提议 → 验证 → 接受或修正”展开：先看候选如何与目标概率对齐，再理解接受概率、residual correction 和 bonus token 分别解决什么问题。最后把这些状态变化串成一轮生成，理解减少逐 token 推进的条件。

**关键词：** `speculative decoding`, `draft model`, `verification`

---
## 前置阅读

**导语：** 先掌握单步解码和 KV Cache 的状态变化，再理解草稿模型如何提出候选、目标模型如何批量验证。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

---

### Step 1: 投机式生成策略如何减少目标模型推进

投机解码让较低成本的草稿模型先提出候选，再由目标模型集中验证。本节聚焦经典的草稿—目标模型协作，核心问题是如何在增加单轮有效推进量的同时保持目标分布不变。

| 环节 | 输入 | 机制责任 | 输出 |
|---|---|---|---|
| 草稿提议 | 当前上下文 | 低成本地产生连续候选及其概率 | 候选 token 与草稿分布 |
| 目标验证 | 候选序列 | 一次计算得到各验证位置的目标分布 | 与候选位置对齐的目标概率 |
| 接受或修正 | 两组对齐概率 | 按接受概率推进，首次拒绝时修正 | 接受前缀与修正 token |
| 全部接受 | 所有候选均通过 | 使用额外目标位置继续推进 | 追加 token |

![Speculative Decoding 流程图](../public/02_PyTorch_Algorithms/23_speculative_decoding_flow.svg)

### Step 2: 草稿候选与目标概率如何对齐

一轮验证需要把草稿 token、草稿概率和目标概率按位置对齐。`K` 表示草稿 token 数，目标模型提供前 `K` 行来验证候选，并额外提供一行用于全部接受时的 bonus token。位置对齐是后续计算接受概率的前提。

| 输入 / 状态 | 形状或内容 | 用途 |
|---|---|---|
| `draft_probs` | `[K, vocab]` | 草稿模型在 K 个位置上的概率分布 |
| `target_probs` | `[K+1, vocab]` | 前 K 行验证草稿，最后一行提供 bonus token |
| `draft_tokens` | 长度为 K 的 token 序列 | 指出每个位置实际提出的候选 |
| 输入检查 | 非负、行归一化、长度和词表维度 | 防止概率与位置错位后继续计算 |

### Step 3: 接受、修正与 bonus 如何决定推进长度
按草稿位置从前到后处理：接受就继续验证，拒绝就从 residual distribution 采样修正 token 并结束本轮；全部 `K` 个草稿都接受时，再从目标模型的 bonus 位置采样一个 token。接受概率决定本轮能推进多长，公式为：

$$\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

接受概率、拒绝修正和全部接受后的追加 token 共同保证输出仍服从目标分布。候选质量决定接受长度，验证与草稿成本则决定这种机制能否带来实际收益。

| 结果 | 接受规则或使用的分布 | 本轮推进 |
|---|---|---|
| 接受 | `p(x) >= q(x)` 时必然接受；否则按 `p(x) / q(x)` 接受 | 进入下一个验证位置 |
| 拒绝 | 对 `max(target - draft, 0)` 归一化后采样 residual correction | 输出修正 token，结束本轮 |
| 全部接受 | 所有草稿 token 通过验证，使用最后一行 `target_probs` | 追加 bonus token |

![Speculative Decoding：验证结果决定下一步](../public/02_PyTorch_Algorithms/23_speculative_acceptance_flow.svg)

### Step 4: 实现并验证单轮投机解码
本题用两组位置对齐的词表概率和候选 token 复现单轮投机解码；输入校验、逐位置遍历和结果结构已经提供，学习者补全接受概率、接受判断、拒绝修正分布和全部接受后的追加采样。测试覆盖非法概率、确定接受、首次拒绝、零概率与全部接受。

| 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| `_accept_probability` | TODO 1：计算候选接受概率 | 处理草稿概率为零，并将比例截断到 1 | 零概率与确定接受测试 |
| `_normalized_residual` | TODO 2：构造拒绝修正分布 | 只保留正概率差并归一化 | 拒绝与零质量测试 |
| `speculative_decode_step` | TODO 3：根据随机样本决定是否接受 | 接受时继续，首次拒绝时立即停止 | 接受数量和检查位置测试 |
| `speculative_decode_step` | TODO 4：全部接受后采样追加 token | 使用额外目标位置而非最后一个验证位置 | 全接受与追加 token 测试 |

```python
import torch
```


```python
def _validate_probability_contract(draft_probs, target_probs, draft_tokens):
    """检查草稿、目标和候选 token 是否能按位置一一对应。"""
    if draft_probs.dim() != 2 or target_probs.dim() != 2:
        raise ValueError('概率张量必须是二维 [K, vocab]')
    K, vocab_size = draft_probs.shape
    if target_probs.shape != (K + 1, vocab_size) or len(draft_tokens) != K:
        raise ValueError('draft/target/token 的长度或词表维度不匹配')
    if not (torch.isfinite(draft_probs).all() and torch.isfinite(target_probs).all()):
        raise ValueError('概率分布不能包含 NaN 或 Inf')
    if (draft_probs < 0).any() or (target_probs < 0).any():
        raise ValueError('概率分布不能包含负数')
    expected_draft = torch.ones(K, device=draft_probs.device, dtype=draft_probs.dtype)
    expected_target = torch.ones(K + 1, device=target_probs.device, dtype=target_probs.dtype)
    if not (torch.allclose(draft_probs.sum(-1), expected_draft, atol=1e-5) and torch.allclose(target_probs.sum(-1), expected_target, atol=1e-5)):
        raise ValueError('输入必须是归一化概率分布')
    if any(not isinstance(token_id, int) or not 0 <= token_id < vocab_size for token_id in draft_tokens):
        raise ValueError('draft token id 必须是词表范围内的整数')
    return K, vocab_size

def _accept_probability(target_prob, draft_prob):
    """返回当前候选的接受概率 alpha。"""
    # TODO 1（接受概率）：处理 q=0，并把 p/q 截断到 [0, 1]。
    # alpha = ???
    raise NotImplementedError('TODO 1：请计算接受概率')

def _normalized_residual(target_row, draft_row):
    """构造拒绝时用于 correction 的归一化 residual 分布。"""
    # TODO 2（拒绝修正）：保留 target-draft 的正概率质量并归一化。
    # residual = ???
    # residual_mass = ???
    raise NotImplementedError('TODO 2：请构造 residual 分布')

def speculative_decode_step(draft_probs, target_probs, draft_tokens, generator=None):
    """
    验证 K 个草稿，并在拒绝或全接受时生成修正 token。
    
    Args:
        draft_probs: 草稿模型在 K 个位置上的概率分布, shape [K, vocab_size]
        target_probs: 目标模型在 K 个验证位置及 1 个 bonus 位置的概率分布, shape [K+1, vocab_size]
        draft_tokens: 草稿模型实际提出的 K 个 token_id, shape [K]
        
    Returns:
        dict: 包含最终 token 序列、接受数量、拒绝状态和检查位置。
    """
    K, vocab_size = _validate_probability_contract(draft_probs, target_probs, draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token_id = draft_tokens[i]
        
        if not 0 <= int(token_id) < vocab_size:
            raise ValueError('draft token id 超出词表范围')
        p = target_probs[i, token_id]
        q = draft_probs[i, token_id]
        
        alpha = _accept_probability(p, q)
        random_draw = torch.rand((), generator=generator).item()
        # TODO 3（接受判断）：比较随机样本与接受概率。
        # accepted = ???
        raise NotImplementedError('TODO 3：请判断当前候选是否接受')
        if accepted:
            accepted_tokens.append(int(token_id))
            continue
        residual = _normalized_residual(target_probs[i], draft_probs[i])
        correction = torch.multinomial(residual, 1, generator=generator).item()
        return {'tokens': accepted_tokens + [correction], 'accepted_count': len(accepted_tokens), 'rejected': True, 'target_positions_checked': i + 1}
    
    # TODO 4（全部接受）：从额外的第 K 行目标分布采样追加 token。
    # bonus = ???
    raise NotImplementedError('TODO 4：请采样全部接受后的追加 token')
    return {'tokens': accepted_tokens + [bonus], 'accepted_count': K, 'rejected': False, 'target_positions_checked': K}


```


```python
# 按机制拆分测试，分别定位输入契约、接受、residual、bonus 和边界状态。
def _build_speculative_fixture():
    draft_tokens = [0, 1, 2]
    draft_probs = torch.zeros(3, 5)
    target_probs = torch.zeros(4, 5)
    draft_probs[0, 0] = 1.0; target_probs[0, 0] = 1.0
    draft_probs[1, 1] = 1.0; target_probs[1, 3] = 1.0
    draft_probs[2, 2] = 1.0; target_probs[2, 2] = 1.0
    target_probs[3, 4] = 1.0
    return draft_tokens, draft_probs, target_probs

def test_probability_contract():
    """输入概率、形状和 token 位置不合法时应明确拒绝。"""
    draft_tokens, draft_probs, target_probs = _build_speculative_fixture()
    invalid = draft_probs.clone()
    invalid[0, 0] = -0.1
    try:
        speculative_decode_step(invalid, target_probs, draft_tokens)
    except ValueError:
        return
    raise AssertionError('负概率应明确被拒绝')

def test_rejection_and_residual():
    """首次拒绝时应从修正分布采样并停止本轮验证。"""
    draft_tokens, draft_probs, target_probs = _build_speculative_fixture()
    result = speculative_decode_step(draft_probs, target_probs, draft_tokens)
    assert result['tokens'] == [0, 3]
    assert result['accepted_count'] == 1 and result['rejected']
    assert result['target_positions_checked'] == 2

def test_all_accept_and_bonus():
    """全部候选接受后应从额外目标位置采样追加 token。"""
    _, _, target_probs = _build_speculative_fixture()
    all_accepted = speculative_decode_step(target_probs[:3], target_probs, [0, 3, 2])
    assert all_accepted['tokens'] == [0, 3, 2, 4]
    assert all_accepted['accepted_count'] == 3
    assert not all_accepted['rejected']

def test_zero_q_acceptance():
    """草稿概率为零而目标概率为正时，候选应被接受。"""
    zero_q_draft = torch.tensor([[0.0, 1.0, 0.0]])
    zero_q_target = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    result = speculative_decode_step(zero_q_draft, zero_q_target, [0])
    assert result['accepted_count'] == 1 and not result['rejected']

def test_zero_residual_failure():
    """拒绝位置没有正修正质量时应明确报错。"""
    draft = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    try:
        speculative_decode_step(draft, target, [1])
    except ValueError:
        return
    raise AssertionError('residual 概率质量为 0 时应明确报错')

def test_speculative_decoding():
    """汇总运行单轮投机解码的机制测试。"""
    try:
        torch.manual_seed(42)
        test_probability_contract()
        test_rejection_and_residual()
        test_all_accept_and_bonus()
        test_zero_q_acceptance()
        test_zero_residual_failure()
        print("✅ 投机解码机制测试通过：输入契约、接受、residual 和 bonus 均通过。")
    except NotImplementedError:
        print("请先完成 TODO 代码。")
        raise
    except Exception as error:
        print(f"❌ 投机解码测试失败: {type(error).__name__}: {error}")
        raise

test_speculative_decoding()

```


## 参考代码与解析

### 代码

```python
def _validate_probability_contract(draft_probs, target_probs, draft_tokens):
    """检查草稿、目标和候选 token 是否能按位置一一对应。"""
    if draft_probs.dim() != 2 or target_probs.dim() != 2:
        raise ValueError('概率张量必须是二维 [K, vocab]')
    K, vocab_size = draft_probs.shape
    if target_probs.shape != (K + 1, vocab_size) or len(draft_tokens) != K:
        raise ValueError('draft/target/token 的长度或词表维度不匹配')
    if not (torch.isfinite(draft_probs).all() and torch.isfinite(target_probs).all()):
        raise ValueError('概率分布不能包含 NaN 或 Inf')
    if (draft_probs < 0).any() or (target_probs < 0).any():
        raise ValueError('概率分布不能包含负数')
    ones_draft = torch.ones(K, device=draft_probs.device, dtype=draft_probs.dtype)
    ones_target = torch.ones(K + 1, device=target_probs.device, dtype=target_probs.dtype)
    if not (torch.allclose(draft_probs.sum(-1), ones_draft, atol=1e-5) and torch.allclose(target_probs.sum(-1), ones_target, atol=1e-5)):
        raise ValueError('输入必须是归一化概率分布')
    if any(not isinstance(token_id, int) or not 0 <= token_id < vocab_size for token_id in draft_tokens):
        raise ValueError('draft token id 必须是词表范围内的整数')
    return K, vocab_size

def _accept_probability(target_prob, draft_prob):
    """返回当前候选的接受概率 alpha。"""
    # TODO 1（接受概率）：处理 q=0，并计算 alpha = min(1, p / q)。
    p, q = float(target_prob), float(draft_prob)
    if q == 0.0:
        return 1.0 if p > 0.0 else 0.0
    return min(1.0, p / q)

def _normalized_residual(target_row, draft_row):
    """构造拒绝时用于 correction 的归一化 residual 分布。"""
    # TODO 2（拒绝修正）：保留 target-draft 的正概率质量并归一化。
    residual = torch.clamp(target_row - draft_row, min=0)
    residual_mass = residual.sum()
    if residual_mass <= 0:
        raise ValueError('residual 概率质量必须大于 0')
    return residual / residual_mass

def speculative_decode_step(draft_probs, target_probs, draft_tokens, generator=None):
    """完成一次 speculative decoding step。

    `target_probs[:K]` 验证草稿，`target_probs[K]` 生成 bonus token。
    """
    K, vocab_size = _validate_probability_contract(draft_probs, target_probs, draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token_id = draft_tokens[i]
        alpha = _accept_probability(target_probs[i, token_id], draft_probs[i, token_id])
        random_draw = torch.rand((), generator=generator).item()
        # TODO 3（接受判断）：比较随机样本与接受概率。
        accepted = random_draw < alpha
        if accepted:
            accepted_tokens.append(int(token_id))
            continue
        residual = _normalized_residual(target_probs[i], draft_probs[i])
        correction = torch.multinomial(residual, 1, generator=generator).item()
        return {'tokens': accepted_tokens + [correction], 'accepted_count': len(accepted_tokens), 'rejected': True, 'target_positions_checked': i + 1}
                
    # TODO 4（全部接受）：从额外的第 K 行目标分布采样追加 token。
    bonus = torch.multinomial(target_probs[K], 1, generator=generator).item()
    return {'tokens': accepted_tokens + [bonus], 'accepted_count': K, 'rejected': False, 'target_positions_checked': K}


```

### 解析

输入契约已经由骨架统一检查，解析只对应学习者需要完成的四项概率机制。

**TODO 1：接受概率**
当草稿概率为正时，将目标概率与草稿概率之比截断到 1；草稿概率为零时单独处理，避免除零。

**TODO 2：拒绝修正分布**
`clamp(target - draft, min=0)` 只保留目标分布尚未被草稿覆盖的概率质量，随后按总质量归一化。

**TODO 3：接受判断**
随机样本小于接受概率时保留候选并继续；首次拒绝会结束当前验证轮，因此判断位置直接决定有效推进长度。

**TODO 4：全部接受后的追加 token**
当 K 个候选全部接受时，从 `target_probs[K]` 采样。该行是额外目标位置，不与任何草稿 token 对齐。

## 相关阅读

投机式生成可从经典接受采样、内部草稿和多候选验证三条路径继续阅读；比较实现时应始终同时检查接受率、验证成本与端到端收益。

- [Speculative Sampling 原论文](https://arxiv.org/abs/2302.01318)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Medusa 原论文：Multiple Decoding Heads](https://arxiv.org/abs/2401.10782)
- [Transformers Assisted Generation 文档](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode.html)
- [35. Native MTP | 原生多 Token 预测](./35_Multi_Token_Decoding.md)
- [INF-EAGLE. Feature Drafter and Candidate Tree | 特征草稿器与候选树](./INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md)
- [INF-PARALLEL-DRAFTING. DFlash and DSpark | 并行草稿与置信度调度](./INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md)
- [68. Speculative Decoding Benchmark | 投机解码基准](./68_Speculative_Decoding_Benchmark.md)