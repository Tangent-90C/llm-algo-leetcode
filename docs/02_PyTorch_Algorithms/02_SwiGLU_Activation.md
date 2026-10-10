# 02. SwiGLU Activation | SwiGLU 激活

**难度：** Easy | **环境：** CPU-first | **标签：** `基础实现`, `激活函数`, `SwiGLU` | **目标人群：** 基础实现学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

Transformer 的 MLP 不只是把向量升维、过激活函数、再降维这么简单。大模型需要在每个 token 的特征里学会“哪些信息该放大，哪些信息该压住”，普通 ReLU/GELU 的单路激活很难显式表达这种选择过程。

SwiGLU 用两条并行分支做门控：一条提供候选特征，另一条决定通过多少。本节沿着单路前馈网络、双路门控和容量对齐展开，帮助你判断门控结构改变了什么，以及为什么中间维需要重新分配。

**关键词：** `SwiGLU`, `GLU`, `gating`

---
## 前置阅读

**导语：** 先能读出逐元素激活、矩阵投影和归一化的位置，再比较 MLP 的单路激活与 SwiGLU 的双路门控。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 14. Activation Functions | 激活函数](../00_Prerequisites/14_Activation_Functions.md)
- [P0: 15. Normalization Techniques | 归一化技术](../00_Prerequisites/15_Normalization_Techniques.md)

---
### Step 1：从单路 FFN 到双路门控

普通 FFN 只有一条非线性变换路径；GLU 增加门控分支，让一组特征控制另一组候选内容；SwiGLU 再把有界的 Sigmoid 门替换为平滑的 SiLU 门，使每个中间特征获得独立的选择强度。

| 结构 | 中间路径 | 选择机制 |
| --- | --- | --- |
| FFN | `up → activation → down` | 所有中间维一起经过同一种激活 |
| GLU | `content × sigmoid(gate) → down` | Sigmoid 门控制候选特征 |
| SwiGLU | `up × SiLU(gate) → down` | SiLU 门保留平滑、可学习的选择 |

![SwiGLU 的 gate / up / down 三条线](../public/02_PyTorch_Algorithms/02_swiglu_gate.svg)
### Step 2：门控容量为什么需要重新对齐

SwiGLU 比单路 FFN 多出一条中间分支。如果仍沿用标准 FFN 的中间宽度，参数量与计算量都会明显增加，因此比较两类结构时要先固定容量口径。常见的 $8d/3$ 中间维来自三次映射与标准 FFN 两次映射之间的参数预算平衡。

| 结构 | 投影数 | 参数量近似 | 与标准 `4d` FFN 对齐时的中间维 |
| --- | ---: | ---: | --- |
| 标准 FFN | 2 | `2dh` | `h = 4d`，约 `8d²` |
| SwiGLU | 3 | `3dh` | `h ≈ 8d/3`，约 `8d²` |

![标准 FFN 与 SwiGLU 的容量对齐](../public/02_PyTorch_Algorithms/02_swiglu_capacity.svg)
### Step 3：门控语义与模型块接口

SwiGLU 的门控值来自 SiLU，因此它不是限制在 0 到 1 之间的概率，也不是简单的开关。门控分支与候选分支必须在同一个中间特征空间逐元素对应；相乘后的结果再回到模型隐藏维，才能继续进入残差路径。

| 机制关系 | 必须保持的约束 | 容易误解之处 |
| --- | --- | --- |
| 门控值 | 由 SiLU 产生，可为负且不受 1 限制 | 不能解释成概率或二值掩码 |
| 两条中间分支 | 特征宽度和逐元素位置必须一致 | 不是两组可以任意配对的特征 |
| MLP 输出 | 恢复为模型隐藏维 | 残差相加与归一化属于模型块的外层关系 |
### Step 4：代码设计与 SwiGLU MLP 契约

题目区依次连接容量推导、硬件倍数对齐、双分支投影和门控回投。固定骨架保留模块接口；学习者补全影响形状与数值结果的四处变量。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| TODO 1 · `calculate_intermediate_size` | 计算 SwiGLU 理论中间维 | 使用 $8/3 \times hidden\_size$ 的容量关系 | 理论维度符合预期 |
| TODO 2 · `calculate_intermediate_size` | 将中间维向上对齐到 `multiple_of` | 对齐后不小于理论维度且是指定倍数 | `4096 → 11008` 等配置正确 |
| TODO 3 · `SwiGLU_MLP.__init__` | 定义融合 gate/up 投影与 down 投影 | 融合投影输出为 `2 × intermediate_size`；均无 bias | 两条分支与回投维度正确 |
| TODO 4 · `SwiGLU_MLP.forward` | 完成 SiLU 门控、逐元素乘法和回投 | `SiLU(gate) * up` 后投回隐藏维 | 输出形状正确，门控与升维分支都影响输出 |

```python
import torch
import torch.nn as nn
```


```python
def calculate_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """计算 LLaMA 风格 SwiGLU 的对齐中间维。"""
    if hidden_size <= 0 or multiple_of <= 0:
        raise ValueError('hidden_size 和 multiple_of 必须为正数')
    # ==========================================
    # TODO 1：按 8/3 容量关系计算理论中间维。
    # ==========================================
    # theoretical_size = ???

    # ==========================================
    # TODO 2：向上对齐到 multiple_of 的倍数。
    # ==========================================
    # aligned_size = ???
    raise NotImplementedError('TODO 1/2：请完成中间维推导与对齐')


class SwiGLU_MLP(nn.Module):
    """使用融合 gate/up 投影的 SwiGLU 前馈层。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError('hidden_size 和 intermediate_size 必须为正数')
        # ==========================================
        # TODO 3：构造无 bias 的融合 gate/up 投影和 down 投影。
        # ==========================================
        # self.gate_up_proj = ???
        # self.down_proj = ???
        raise NotImplementedError('TODO 3：请定义 SwiGLU 投影层')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 gate/up 切分、SiLU 门控和 down 投影。"""
        # ==========================================
        # TODO 4：完成门控前向。
        # ==========================================
        # gate_up = ???
        # gate, up = ???
        # gated = ???
        raise NotImplementedError('TODO 4：请完成 SwiGLU 门控前向')

```


```python
# 测试设计：分别验证容量推导、投影契约与门控前向，而非只检查最终形状。

def test_swiglu_intermediate_size():
    """验证理论中间维、倍数对齐和非法输入拒绝。"""
    assert calculate_intermediate_size(4096, multiple_of=256) == 11008
    assert calculate_intermediate_size(64, multiple_of=16) == 176
    for hidden_size, multiple_of in ((0, 16), (64, 0)):
        try:
            calculate_intermediate_size(hidden_size, multiple_of)
        except ValueError:
            continue
        raise AssertionError('非法维度参数应被拒绝')


def test_swiglu_projection_contract():
    """验证融合投影与回投影的形状和 bias 契约。"""
    mlp = SwiGLU_MLP(hidden_size=8, intermediate_size=16)
    assert mlp.gate_up_proj.in_features == 8
    assert mlp.gate_up_proj.out_features == 32
    assert mlp.down_proj.in_features == 16
    assert mlp.down_proj.out_features == 8
    assert mlp.gate_up_proj.bias is None and mlp.down_proj.bias is None


def test_swiglu_gated_forward():
    """验证门控前向数值、输出形状和梯度连通性。"""
    torch.manual_seed(7)
    mlp = SwiGLU_MLP(hidden_size=4, intermediate_size=6)
    x = torch.randn(2, 3, 4, requires_grad=True)
    output = mlp(x)
    gate, up = torch.chunk(mlp.gate_up_proj(x), 2, dim=-1)
    expected = mlp.down_proj(torch.nn.functional.silu(gate) * up)
    assert output.shape == x.shape
    assert torch.allclose(output, expected)
    output.sum().backward()
    assert x.grad is not None


def run_swiglu_tests():
    """运行 SwiGLU 的机制测试集合。"""
    test_swiglu_intermediate_size()
    test_swiglu_projection_contract()
    test_swiglu_gated_forward()
    print('✅ SwiGLU：容量、投影与门控前向测试通过')


run_swiglu_tests()

```

🛑 **STOP HERE** 🛑

请先完成题目区并通过测试，再继续查看参考代码与解析。
## 参考代码与解析

### 代码

```python
def calculate_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """计算 LLaMA 风格 SwiGLU 的对齐中间维。"""
    if hidden_size <= 0 or multiple_of <= 0:
        raise ValueError('hidden_size 和 multiple_of 必须为正数')
    # TODO 1：按 8/3 容量关系计算理论中间维。
    theoretical_size = int(hidden_size * 8 / 3)
    # TODO 2：向上对齐到 multiple_of 的倍数。
    aligned_size = ((theoretical_size + multiple_of - 1) // multiple_of) * multiple_of
    return aligned_size


class SwiGLU_MLP(nn.Module):
    """使用融合 gate/up 投影的 SwiGLU 前馈层。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError('hidden_size 和 intermediate_size 必须为正数')
        # TODO 3：构造无 bias 的融合 gate/up 投影和 down 投影。
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 gate/up 切分、SiLU 门控和 down 投影。"""
        # TODO 4：完成门控前向。
        gate_up = self.gate_up_proj(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        gated = torch.nn.functional.silu(gate) * up
        return self.down_proj(gated)

```

### 解析

#### TODO 1：推导理论中间维

- 判断依据：SwiGLU 有 gate、up、down 三个投影，需要在相近参数预算下与标准 `4D` FFN 比较。
- 实现与证据：按 $8/3 \times hidden\_size$ 得到理论宽度，测试检查已知配置。

#### TODO 2：执行倍数对齐

- 判断依据：对齐值必须不小于理论宽度，并能被 `multiple_of` 整除。
- 实现与证据：使用向上取整而非四舍五入，测试覆盖常见维度和非法参数。

#### TODO 3：建立三投影契约

- 判断依据：融合投影一次产生等宽的 gate/up 两支，down 投影再回到模型隐藏维。
- 实现与证据：测试核对输入输出维度及无 bias 约束。

#### TODO 4：完成门控数据流

- 判断依据：候选内容必须经过 `SiLU(gate) * up` 选择后才能回投。
- 实现与证据：测试同时核对参考数值、输出形状和反向梯度。
## 相关阅读

本节的门控激活可以继续沿两条线阅读：一条是 GLU 家族的原始设计，另一条是 LLaMA 实现中 gate / up / down 三个投影如何落地。

- [GLU 家族原论文：GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Transformers 中的 LLaMA 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [03. RoPE 旋转位置编码](../02_PyTorch_Algorithms/03_RoPE_Tutorial.md)
- [04. 多头注意力与 GQA](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [P1: Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
