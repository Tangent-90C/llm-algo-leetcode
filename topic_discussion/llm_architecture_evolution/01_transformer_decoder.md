# 01 仅解码器主干（Decoder-only Backbone）

## 学习问题

现代大语言模型通常把词元表示送入重复堆叠的解码器块，并用因果掩码约束每个位置只能读取已经出现的词元。归一化、注意力、MLP、残差与缓存都依附于这条主干的数据流和状态接口。

## 核心机制

### 设计压力：为什么主干收敛到仅解码器结构？

编码器—解码器适合“先编码输入、再条件生成”的任务；自回归语言建模只需一条按词元向前推进的状态路径。仅解码器结构让预训练目标、推理循环和上下文缓存共享同一结构接口。

| 结构 | 信息路径 | 典型目标 | 主要代价 |
|:---|:---|:---|:---|
| 编码器—解码器 | 编码器表示经交叉注意力进入解码器 | 翻译、条件生成 | 两套主干与跨模块接口更复杂 |
| 仅解码器 | 所有词元沿因果路径进入同一组 Transformer 块 | 下一词元预测、生成 | 长上下文注意力与状态成本持续增长 |

![仅解码器主干：词元表示经过多个 Transformer 块形成下一词元分布](../../docs/public/topic_discussion/llm_architecture_evolution/block_overview.svg)

### 结构机制：一个 Transformer 块如何更新主干状态？

每个 Transformer 块都围绕同一条残差流完成两次更新：注意力负责词元间交互，MLP 负责逐词元的通道变换；归一化控制送入分支的尺度，残差连接保留并累积主干状态。多个块重复该结构，形成模型深度。

同一组参数在训练与生成时复用，但状态生命周期不同：训练通常一次处理整段序列并保存反向传播所需激活；生成先执行预填充，再逐词元解码并持续追加 KV Cache。架构审计必须把“模型包含哪些模块”与“执行阶段保存哪些状态”分开记录。

| 组件 | 结构职责 | 训练状态 | 生成状态 | 后续演进问题 |
|:---|:---|:---|:---|:---|
| 归一化 + 残差 | 稳定深层状态更新 | 激活与反向所需中间量 | 当前词元的主干状态 | Pre-Norm、RMSNorm、残差缩放 |
| 注意力 | 读取上下文并混合词元信息 | Q/K/V 与注意力中间量 | 历史 KV 或递推状态 | MHA/GQA/MLA、稀疏访问、混合记忆 |
| MLP / FFN | 重组单个词元的通道表示 | 门控与中间激活 | 当前步中间激活 | SwiGLU、MoE、专家稀疏激活 |
| 输出头 | 将隐状态映射到词表分布 | 词元级训练目标 | 下一词元 logits | 权重共享、采样与生成循环 |

## 选择条件与验证证据

“采用仅解码器结构”只能确定主干范式，不能直接推出训练稳定性、长上下文质量或推理速度。结构审计还需要记录 Transformer 块数量、隐藏维度、归一化位置、注意力表示、MLP 类型、上下文长度以及执行后端，再用对应证据验证质量和成本。

先用 [Part 02 · 00 PyTorch 热身](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb)建立最小接口，再在 [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)组装两次残差更新；配置审计进入 [Part 02 · 08](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb)，多层稳定性对照进入 [Part 02 · 60](../../02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark.ipynb)。

## 相关阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
