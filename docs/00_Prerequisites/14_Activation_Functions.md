# 14. Activation Functions | 激活函数

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `非线性`, `门控激活` | **目标人群：** 准备理解 MLP 与门控结构的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/14_Activation_Functions.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


激活函数同时改变前向输出和反向梯度。ReLU 引入明确截断区，GELU 与 SiLU 使用平滑门控；现代 LLM 的 MLP 还会把 SiLU 放进 SwiGLU 等门控结构。本页从输出、梯度和门控三条线建立可计算的比较。

在 Transformer Block 中，激活通常位于 MLP 分支；它决定哪些特征继续传播，也影响 backward 保存的中间状态。

**关键词：** `ReLU`, `GELU`, `SiLU`, `SwiGLU`, `gradient`

![激活函数的输出、梯度与门控关系](../public/00_Prerequisites/14_activation_gradient_map.svg)

## 前置阅读
**导语：** Tensor 运算给出前向值，Autograd 给出局部梯度；两者共同决定激活函数的行为。
- [05. PyTorch Tensor Fundamentals | PyTorch 张量基础](./05_PyTorch_Tensor_Fundamentals.md)
- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](./07_PyTorch_Autograd_and_Backward.md)
- [0D 组页](./0D.md)

## 相关阅读
**导语：** 继续观察归一化如何控制 MLP 输入尺度，以及激活如何进入完整 Transformer Block。
- [15. Normalization Techniques | 归一化技术](./15_Normalization_Techniques.md)
- [Part 02 · 05 LLaMA3 Block](../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)

## Q1：激活函数在模型里解决什么问题？

没有激活函数，多层线性变换仍可合并成一个线性变换。ReLU、GELU 和 SiLU 都引入非线性，但它们对负值、零点附近和大正值的处理不同。


```python
import math
import torch
import torch.nn.functional as F


def relu(x):
    return torch.clamp(x, min=0)


def gelu_exact(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x):
    return x * torch.sigmoid(x)


def activation_summary(x):
    return {
        'relu_mean': float(relu(x).mean().item()),
        'gelu_mean': float(gelu_exact(x).mean().item()),
        'silu_mean': float(silu(x).mean().item()),
    }


x = torch.linspace(-3, 3, steps=7)
print('输入：', x.tolist())
print('ReLU：', relu(x).tolist())
print('GELU：', [round(v, 4) for v in gelu_exact(x).tolist()])
print('SiLU：', [round(v, 4) for v in silu(x).tolist()])

```

## Q1 验证：不同激活的输出形态是否可见？

这里直接把同一组输入喂给不同激活，看看它们对负值、零值和正值的处理是否符合预期。


```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
assert torch.allclose(relu(x), F.relu(x))
assert torch.allclose(gelu_exact(x), F.gelu(x), atol=1e-6, rtol=1e-6)
assert torch.allclose(silu(x), F.silu(x))
print('✅ 基本激活函数通过')

```

## Q2：输出形状相似时，梯度行为有什么不同？

激活函数不仅改变输出，也改变局部导数。ReLU 在负半轴的梯度为零；GELU 与 SiLU 在零点附近保留平滑变化。比较同一输入上的输出和梯度，比只记公式更能解释训练行为。


```python
def value_and_gradient(function, values):
    """返回激活输出以及输出和对输入的梯度。"""
    inputs = values.clone().detach().requires_grad_(True)
    outputs = function(inputs)
    outputs.sum().backward()
    return outputs.detach(), inputs.grad.detach()


points = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
for name, function in {'relu': relu, 'gelu': gelu_exact, 'silu': silu}.items():
    values, gradients = value_and_gradient(function, points)
    print(name, 'values=', values.tolist(), 'gradients=', gradients.tolist())

```

## Q2 验证：截断区与平滑区是否可观察？

检查 ReLU 负半轴梯度为零，同时确认 GELU 与 SiLU 的输出和梯度均保持有限。


```python
relu_values, relu_gradients = value_and_gradient(relu, points)
gelu_values, gelu_gradients = value_and_gradient(gelu_exact, points)
silu_values, silu_gradients = value_and_gradient(silu, points)
assert torch.equal(relu_gradients[:2], torch.zeros(2))
assert torch.isfinite(gelu_values).all() and torch.isfinite(gelu_gradients).all()
assert torch.isfinite(silu_values).all() and torch.isfinite(silu_gradients).all()
assert not torch.equal(gelu_gradients[:2], torch.zeros(2))
print('✅ 激活输出与梯度通过')

```

## Q3：SwiGLU 如何把激活函数变成门控分支？

门控 MLP 不只是把激活接在线性层后面，而是让一条分支生成门值，再逐元素调制另一条内容分支。SwiGLU 使用 SiLU 处理 gate，因此输入通常需要产生两组同形状张量。


```python
def swiglu(gate, value):
    """使用 SiLU 门控 value 分支；两个输入必须同形状。"""
    if gate.shape != value.shape:
        raise ValueError('gate and value must have the same shape')
    return silu(gate) * value


gate = torch.tensor([[-2.0, 0.0, 2.0]])
value = torch.tensor([[3.0, 3.0, 3.0]])
print('gate：', silu(gate))
print('SwiGLU 输出：', swiglu(gate, value))

```

## Q3 验证：门控分支是否按元素调制内容？

零 gate 应关闭对应位置，正负 gate 则产生不同缩放；shape 不一致必须立即报错。


```python
expected = F.silu(gate) * value
assert torch.allclose(swiglu(gate, value), expected)
assert swiglu(gate, value)[0, 1].item() == 0.0
try:
    swiglu(torch.ones(2, 3), torch.ones(2, 4))
except ValueError:
    pass
else:
    raise AssertionError('shape mismatch should fail')
print('✅ SwiGLU 门控契约通过')

```

## Q4：怎样用同一组统计比较激活行为？

比较激活时需要同时记录输出尺度、零值比例和梯度尺度。单看均值会漏掉 ReLU 的稀疏性，也无法看出平滑激活是否保留了负半轴梯度。


```python
def activation_profile(function, values):
    """汇总激活输出与局部梯度，不把统计量转换成预设标签。"""
    outputs, gradients = value_and_gradient(function, values)
    return {
        'output_mean': float(outputs.mean()),
        'output_std': float(outputs.std(unbiased=False)),
        'zero_fraction': float(outputs.eq(0).float().mean()),
        'gradient_abs_mean': float(gradients.abs().mean()),
        'gradient_abs_max': float(gradients.abs().max()),
    }


sample = torch.linspace(-4, 4, steps=17)
profiles = {name: activation_profile(function, sample) for name, function in {
    'relu': relu,
    'gelu': gelu_exact,
    'silu': silu,
}.items()}
for name, profile in profiles.items():
    print(name, profile)

```

## Q4 验证：统计字段是否能区分截断与平滑门控？

ReLU 应出现明显零值比例；GELU 与 SiLU 在该输入上不应产生同样的硬截断。所有统计必须保持有限。


```python
assert profiles['relu']['zero_fraction'] > 0.4
assert profiles['gelu']['zero_fraction'] < profiles['relu']['zero_fraction']
assert profiles['silu']['zero_fraction'] < profiles['relu']['zero_fraction']
for profile in profiles.values():
    assert all(torch.isfinite(torch.tensor(value)) for value in profile.values())
print('✅ 激活统计对照通过')

```
