# 21. Decoding Strategies | 解码策略
**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `解码`, `Sampling` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

模型生成下一个 token 时，会先为整张词表打分，再从候选中选择一个结果。永远选择最高分比较稳定，但可能缺少变化；完全随机又难以控制。解码策略要解决的，就是如何根据分数分布和选择规则，在确定性与多样性之间作出可解释的取舍。

学习时先用 greedy 建立确定性基线，再观察 temperature 如何改变分布形状、top-k / top-p 如何改变候选集合，最后把候选数量、熵和随机性放在一起比较。这样可以从“分数如何变”逐步理解“候选如何选”和“结果为何不同”。

**关键词：** `top-k`, `top-p`, `temperature`

---

## 前置阅读

**导语：** 先回顾注意力模块如何产生当前位置的表示，以及模型如何将表示映射为词表 logits；然后观察不同解码规则怎样改变候选集合和采样结果。
- [04. Attention MHA GQA | 注意力机制：MHA、MQA、GQA](./04_Attention_MHA_GQA.md)
- [可选前置：05. LLaMA3 Block Tutorial | LLaMA3 Block 实现](./05_LLaMA3_Block_Tutorial.md)

---

### Step 1: 从词表分数到下一个 token
模型在每个生成位置输出一组词表 logits。解码的输入是这组分数和选择配置，处理过程可以改变分布形状、缩小候选范围，最后输出一个下一个 token：`greedy` 取最高分，sampling 从重整化后的概率中抽样。

| 输入或路径 | 主要处理 | 输出与观察重点 |
|---|---|---|
| 原始 logits | 为每个 token 提供分数 | 候选排序与分布形状 |
| greedy | 直接选择最高分 token | 结果确定，便于作为基线 |
| sampling | 根据重整化概率抽样 | 结果具有多样性和随机性 |
| Top-K / Top-p | 在 sampling 前缩小候选集合 | 候选空间和概率质量发生变化 |

![解码策略流程](../public/02_PyTorch_Algorithms/21_decoding_pipeline.svg)

### Step 2: 候选处理顺序如何改变分布
把最后一个位置的 logits 依次处理，再进入确定性选择或概率采样。本节采用 Temperature → Top-K → Top-p → Softmax → 选择的顺序：前两步改变分数或候选范围，Softmax 之后才得到用于 sampling 的概率。temperature 很小时仍可能采样到其他 token，不能把它当作 greedy。

| 阶段 | 输入 | 输出 / 作用 | 观察重点 |
|---|---|---|---|
| Temperature | 原始 logits、温度 $T$ | 缩放后的 logits | 排序不变，分布尖锐程度改变 |
| Top-K | 缩放后的 logits、$K$ | 只保留前 K 个候选 | 固定数量；相同分数可能保留超过 K 个 |
| Top-p | 候选 logits、$p$ | 保留累计概率达到阈值的最小集合 | 必须保留首次达到阈值的边界 token |
| 选择 | 重整化后的概率或 logits | 一个下一个 token | greedy 取最大值，sampling 按概率选择 |

### Step 3: Top-p 如何确定边界并重整化概率

Top-p 的关键是保留累计概率首次达到或超过阈值的边界 token。假设排序后的概率为 `[0.5, 0.3, 0.1, 0.05, 0.05]`，当 $p=0.85$ 时，累计概率在第三个 token 首次超过阈值，因此前三个 token 进入候选集合。

过滤后，候选之外的 logits 设为 `-inf`，再对保留的 logits 做一次 Softmax；这样采样概率会在新的候选集合内重新归一化。

| 处理步骤 | 作用 | 示例或检查点 |
|---|---|---|
| 排序与累加 | 按概率从高到低计算 `cumsum` | `[0.5, 0.3, 0.1, 0.05, 0.05]` → `[0.5, 0.8, 0.9, 0.95, 1.0]` |
| 保留边界 | 保留累计概率首次达到或超过 $p$ 的 token | $p=0.85$ 时保留前三个 token，边界 token 不能被误删 |
| 屏蔽候选 | 将边界之后的 logits 设为 `-inf` | 被屏蔽 token 不再参与选择 |
| 重整化 | 对保留 logits 再执行 Softmax | 候选概率重新归一化为 1 |

![Top-p 边界保留示意](../public/02_PyTorch_Algorithms/21_top_p_boundary.svg)
### Step 4: 实现候选过滤并验证解码流程

题目区提供 token 选择和逐步生成骨架。学习者完成温度缩放、Top-K 阈值过滤和 Top-p 边界恢复三个机制；测试分别覆盖非法温度、同分候选、累计概率边界、确定性选择、可复现采样与 batch 输入。

| 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| `apply_temperature` | 校验温度并缩放 logits | 保持排序、shape、dtype 与 device | 正常缩放与非法温度 |
| `apply_top_k` | 按第 K 大值过滤候选 | 同分候选一起保留，其余位置为 `-inf` | 普通候选、ties 与 batch |
| `apply_top_p` | 保留累计概率边界并恢复词表顺序 | 首次达到阈值的 token 不能被删除 | 边界集合、非法阈值与 batch |


```python
import torch
import torch.nn.functional as F
```


```python
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """在 Softmax 前缩放 logits。

    参数：
        logits: 最后一维为词表维度的分数张量，支持单样本或 batch。
        temperature: 有限正数；越小通常使分布更尖锐。
    """
    # ==========================================
    # TODO 1: 检查 temperature 并完成缩放
    # 要求：temperature 必须是有限正数；输出保持 logits 的 shape、dtype 和 device，
    #       只改变分数尺度，不改变 token 的排序关系。
    # ==========================================
    # if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
    #     raise ValueError('temperature 必须是有限正数')
    # temp = ???  # 有限正数的 temperature
    raise NotImplementedError("TODO 1：请校验并使用 temperature")
    return logits / temp

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """按第 K 大值筛选 logits，保留 ties 是本实现的明确语义。

    `top_k=0` 或不小于词表大小时保持输入不变。
    """
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k 必须为非负整数')
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
        
    # ==========================================
    # TODO 2: Top-K 截断
    # 先找到第 K 大值作为阈值，再保留所有不小于阈值的位置；ties 可能使保留数超过 K。
    # 输出仍为 [..., vocab]，被过滤位置改为 -inf。
    # ==========================================
    # filter_value = ???  # 被过滤位置使用的 -inf
    # kth_values = ???  # 第 K 大阈值，形状保留最后一维
    # logits = ???  # 小于阈值的位置置为 filter_value
    return logits

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """按累计概率保留最小候选集合，并恢复原始词表顺序。

    边界 token 会被保留；因此这里的 top-p 是“首次达到阈值”的语义。
    """
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError('top_p 必须满足 0 < top_p <= 1')
        
    # 1. 首先需要将 logits 从大到小排序
    # 注意我们需要记住原始的索引 (indices)，因为截断完了还要把它复原回原来的位置！
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 对排序后的 logits 做 Softmax，再计算累加概率
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p

    # ==========================================
    # TODO 3: 保留边界 token，并恢复原词表顺序
    # 变量：previous_mask 是右移前的掩码副本；restored_logits 是最终输出。
    # 提示：将掩码右移一位，首位设为 False；屏蔽后再用 sorted_indices 恢复顺序。
    # ==========================================
    # previous_mask = ???
    # restored_logits = ???
    sorted_indices_to_remove[..., 1:] = previous_mask
    sorted_indices_to_remove[..., 0] = False
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return restored_logits

def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """完成一次候选过滤和 token 选择，支持 greedy 与可复现 sampling。

    输入可以是 `[vocab]` 或 `[batch, vocab]`；内部统一为二维，输出统一为 `[batch, 1]`。
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError('logits 必须是 [vocab] 或 [batch, vocab]')
    # 1. 调温
    logits = apply_temperature(logits, temperature)
    
    # 2. Top-K 截断 (通常先 K 后 p)
    logits = apply_top_k(logits, top_k)
    
    # 3. Top-p 截断
    logits = apply_top_p(logits, top_p)
    
    # 4. 概率重归一化
    probs = F.softmax(logits, dim=-1)
    
    # 5. greedy 直接取最大值；sampling 才使用 multinomial
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token

def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """用 logits_fn 演示逐 token 解码；不代表真实模型性能。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens

```


```python
# 测试设计：分别验证温度、Top-K、Top-p、选择循环与输入契约。

def _fixture_logits():
    return torch.tensor([[0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]])

def test_temperature_contract():
    logits = _fixture_logits()
    scaled = apply_temperature(logits, 0.5)
    assert torch.allclose(scaled[0, 5] - scaled[0, 6], (logits[0, 5] - logits[0, 6]) * 2)
    assert scaled.shape == logits.shape and scaled.dtype == logits.dtype
    assert torch.equal(torch.argsort(logits), torch.argsort(scaled))
    try:
        apply_temperature(logits, 0.0)
    except ValueError:
        return
    raise AssertionError('非法 temperature 未被拒绝')

def test_top_k_filtering():
    logits = _fixture_logits().to(dtype=torch.float64)
    filtered = apply_top_k(logits, 3)
    assert torch.isfinite(filtered).sum().item() == 3
    assert filtered.shape == logits.shape and filtered.dtype == logits.dtype and filtered.device == logits.device
    tied = torch.tensor([[2.0, 2.0, 2.0, 1.0]])
    assert torch.isfinite(apply_top_k(tied, 2)).sum().item() == 3

def test_top_p_boundary():
    logits = _fixture_logits().to(dtype=torch.float64)
    filtered = apply_top_p(logits, 0.8)
    assert torch.isfinite(filtered).sum().item() == 3
    assert torch.equal(filtered[0, [5, 6, 1]], logits[0, [5, 6, 1]])
    assert filtered.shape == logits.shape and filtered.dtype == logits.dtype and filtered.device == logits.device
    for invalid_p in (0.0, 1.1):
        try:
            apply_top_p(logits, invalid_p)
        except ValueError:
            continue
        raise AssertionError('非法 top_p 未被拒绝')

def test_selection_and_autoregressive_loop():
    logits = _fixture_logits()
    assert decode_next_token(logits, 1.0, 0, 1.0, do_sample=False).item() == 5
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(decode_next_token(logits, generator=g1), decode_next_token(logits, generator=g2))
    prompt = torch.tensor([[1, 2]])
    def fake_logits(tokens):
        result = torch.zeros(tokens.size(0), tokens.size(1), 10)
        result[..., 3] = 2.0
        return result
    generated = autoregressive_decode(prompt, fake_logits, 3, temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
    assert generated.shape == (1, 5) and torch.equal(generated[0, -3:], torch.tensor([3, 3, 3]))

def test_batch_and_input_contract():
    logits = _fixture_logits().repeat(2, 1)
    assert apply_top_k(logits, 3).shape == logits.shape
    assert apply_top_p(logits, 0.8).shape == logits.shape
    assert decode_next_token(logits[0], 1.0, 0, 1.0, do_sample=False).shape == (1, 1)

def run_decoding_tests():
    test_temperature_contract()
    test_top_k_filtering()
    test_top_p_boundary()
    test_selection_and_autoregressive_loop()
    test_batch_and_input_contract()
    print('✅ 解码机制测试通过：温度、候选过滤、选择循环与输入契约均符合预期。')

run_decoding_tests()

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
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """在 Softmax 前缩放 logits；temperature 必须是有限正数。"""
    # TODO 1: 校验 temperature，不要用静默截断替代非法参数处理
    # 要求：必须是有限正数；输出保持 logits 的 shape、dtype 和 device，只改变分数尺度。
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise ValueError('temperature 必须是有限正数')
    temp = temperature
    return logits / temp

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K 截断；按第 K 大值筛选，分数相同时可能保留多个 token。
    """
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k 必须为非负整数')
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
        
    # TODO 2: 实现 Top-K 截断
    # 先找到第 K 大值作为阈值，再保留所有不小于阈值的位置；ties 可能使保留数超过 K。
    filter_value = float('-inf')
    
    # 找到第 K 大的值
    kth_values, _ = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
    kth_values = kth_values[..., -1:] # 取最后一个（第 K 大的值）
    
    # 将小于第 K 大值的位置设为 -inf
    logits = torch.where(logits < kth_values, torch.full_like(logits, filter_value), logits)
    return logits

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p 截断，保留累计概率首次达到阈值的最小候选集合。
    """
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError('top_p 必须满足 0 < top_p <= 1')
        
    # 1. 首先需要将 logits 从大到小排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 对排序后的概率计算累加和
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # TODO 3: 保留边界 token，并恢复原词表顺序
    # previous_mask 保存右移前的掩码；restored_logits 是恢复后的最终输出。
    sorted_indices_to_remove = cumulative_probs > top_p
    
    previous_mask = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 1:] = previous_mask
    sorted_indices_to_remove[..., 0] = False
    
    # 将需要剔除的 sorted_logits 设为极小值
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # 将排序后的 logits 恢复到原始顺序
    restored_logits = torch.zeros_like(logits).scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
    
    return restored_logits

def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """应用策略并选择一个 token；do_sample=False 时使用 greedy。

    输入可以是 `[vocab]` 或 `[batch, vocab]`；内部统一为二维，输出统一为 `[batch, 1]`。
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError('logits 必须是 [vocab] 或 [batch, vocab]')
    # 1. 调温
    logits = apply_temperature(logits, temperature)
    
    # 2. Top-K 截断 (通常先 K 后 p)
    logits = apply_top_k(logits, top_k)
    
    # 3. Top-p 截断
    logits = apply_top_p(logits, top_p)
    
    # 4. 概率重归一化
    probs = F.softmax(logits, dim=-1)
    
    # 5. greedy 与 sampling 是两条不同路径
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token

def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """用一个 logits_fn 演示逐 token 解码；不代表真实模型性能。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens
```

### 解析

**1. TODO 1：Temperature 温度调节**
- 先确认 `temperature` 为有限正数，再执行 `logits / temperature`。温度改变分布尖锐程度，不改变 logits 排序；greedy 则由 `do_sample=False` 选择。

**2. TODO 2：Top-K 截断**
- 用第 K 大 logits 作为阈值，保留不低于阈值的位置并将其余位置设为 `-inf`。同分 token 会一起保留，因此候选数可能超过 K。

**3. TODO 3：Top-p 核采样**
- 在排序空间计算累计概率，右移屏蔽掩码以保留首次达到阈值的边界 token，再用 `scatter_` 恢复原词表顺序。函数返回过滤后的 logits；采样前的 Softmax 会在保留集合内重新归一化。

**把机制接入真实生成**
- 本实现采用 Temperature → Top-K → Top-p；换序会改变候选集合。
- `-float('inf')` 屏蔽候选，同时保持张量形状；最终 Softmax 会在保留集合内重新归一化。
- 接入模型时，可用相同配置比较 greedy 与 sampling 的输出差异；23、35 和 36 再分别处理候选验证、批量推进与解码调度。
## 相关阅读

完成本节的候选过滤和采样实现后，可以沿着“采样方法 → 真实生成接口 → 解码服务策略”继续阅读。

- [The Curious Case of Neural Text Degeneration：Nucleus Sampling 原论文](https://arxiv.org/abs/1904.09751)
- [Attention Is All You Need 原论文](https://arxiv.org/abs/1706.03762)
- [Transformers 文本生成策略文档](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [vLLM SamplingParams 文档](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html)
- [Part 02 · 23 投机解码](./23_Speculative_Decoding.md)
- [Part 02 · 35 多 Token 解码](./35_Multi_Token_Decoding.md)
- [Part 02 · 36 解码调度](./36_Decode_Scheduling.md)
