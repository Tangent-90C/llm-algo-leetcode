# 08 代表模型与结构审计（Representative Models and Architecture Audit）

## 学习问题

面对一个真实模型，需要把型号名称还原为可核对的结构字段，并区分基础架构变化、后训练能力变化与运行时实现差异。判断依据来自固定版本的配置、模块树、权重形状和实际运行路径。

## 核心机制

### 从模型名称回到结构字段

模型家族只能提供检索入口，不能代替结构结论。一次结构审计至少要核对输入表示、Transformer 块组织、注意力表示、位置机制、FFN 类型、记忆接口和配置是否真正实例化。

| 审计层 | 需要核对的字段 | 常见误判 |
|:---|:---|:---|
| 输入表示 | 分词器、词表、嵌入、权重绑定 | 把词表变化误写成 Transformer 块变化 |
| Transformer 块主干 | 仅解码器、前置归一化、残差顺序、层数 | 只看模型名称推断残差路径 |
| 注意力 | MHA/MQA/GQA/MLA、KV 注意力头、访问范围 | 把 KV 压缩与稀疏访问混为一谈 |
| 位置机制 | RoPE 参数、缩放、训练长度与目标长度 | 只凭声明长度判断可用上下文 |
| FFN 与稀疏性 | 稠密 FFN、SwiGLU、MoE、共享专家、Top-k | 只看总参数量推断每个词元的成本 |
| 记忆接口 | 完整 KV、局部 KV、递推状态、混合层 | 把 SSM 直接称为注意力变体 |
| 实例化结果 | 配置、模块树、权重形状、运行路径 | 把配置声明当成实际执行证据 |

### 用代表模型建立参照系

代表模型用于观察结构组合，不用于给模型家族贴固定标签。同一家族会跨代改变注意力、MoE、上下文和部署策略，最终判断必须回到固定版本的配置与权重。

| 参照系 | 适合观察的结构问题 | 本专题入口 |
|:---|:---|:---|
| LLaMA 类稠密解码器 | RMSNorm、RoPE、SwiGLU 与标准 Transformer 块如何组合 | 正文 01、03、05–07 |
| Mistral 类长上下文模型 | 局部/窗口访问如何改变长序列成本 | 正文 04、05；Part 02 · 44 |
| Qwen 稠密 / MoE 家族 | 输入覆盖、稠密 / MoE 分支与工程配置如何共同变化 | 正文 02、09；Part 02 · 06–08 |
| DeepSeek V 系列 | MLA、MoE 与稀疏访问如何重写状态和执行成本 | 正文 04、09；Part 02 · ARCH-MLA-KV |
| 推理 / 对话派生模型 | 后训练和推理模式是否改变基础 Transformer 块 | 后训练优化专题；不得仅凭能力名称判定架构变化 |

![代表模型结构矩阵：从模型名回到输入、Transformer 块、注意力、FFN 与记忆接口](../../docs/public/topic_discussion/llm_architecture_evolution/representative_models_matrix.svg)

## 选择条件与验证证据

结构审计按“声明—实例化—运行”三层取证。只完成其中一层时，结论必须停留在对应证据等级。

| 证据层 | 检查对象 | 可以得出的结论 |
|:---|:---|:---|
| 声明层 | 模型卡、论文、`config.json` | 模型声称采用哪些结构与参数 |
| 实例化层 | 模块树、权重名称与形状、加载日志 | 当前模型快照实际包含哪些模块 |
| 运行层 | 性能分析记录、状态大小、执行后端路径、输出质量 | 目标环境中真正执行了什么，代价是否符合预期 |

审计时固定模型版本、分词器版本、数据类型和执行后端；将关键结构字段、来源、冲突项和未验证项写入同一份记录。基础模型与后训练派生模型应分别登记，避免把能力变化错误归因于 Transformer 块变化。

- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb)：读取配置、模块树与结构不变量。
- [Part 02 · 61 模型架构审计](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)：汇总声明、实例化与运行证据。
- [Part 02 · 71 MLA 与 KV 表示项目](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb)：核对注意力表示与状态成本。
- [Part 02 · 77 长序列记忆架构基准](../../02_PyTorch_Algorithms/77_Long_Sequence_Memory_Architecture_Benchmark.ipynb)：比较显式 KV、递推状态和混合记忆。
- [Part 02 · 78 注意力访问模式项目](../../02_PyTorch_Algorithms/78_Attention_Access_Pattern_Benchmark.ipynb)：验证访问范围与长度代价。

## 相关阅读

- [LLaMA](https://arxiv.org/abs/2302.13971)：现代稠密解码器结构参照。
- [Mistral 7B](https://arxiv.org/abs/2310.06825)：窗口访问与长上下文效率参照。
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)：多语言与工程配置参照。
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)：稠密与 MoE 双路线参照。
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)：MLA 与 MoE 结构参照。
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：更大规模 MoE 系统的结构参照。
