# 02. Prefill and Attention Kernel | 预填充与注意力计算内核

## 学习问题

长输入会增加预填充计算、注意力工作集和 KV Cache 写入量，但 TTFT 升高不一定都来自同一处。沿着“输入词元 → 注意力计算 → KV 状态 → 执行路径”检查，才能区分 FlashAttention、分块预填充和前缀缓存分别改变什么。

学完后，你应能根据阶段时间、工作集、实际算子路径和请求指标选择候选，而不是只凭 API 名称或一次 TTFT 变化判断收益。

## 核心机制

### 预填充如何建立注意力与 KV 状态

模型类型、注意力结构和持久状态会改变预填充的张量形状、工作集和执行路径。预填充一次处理已有输入。每层先生成 Q、K、V，计算 Q 与 K 的相关性并归一化，再聚合 V；本层产生的 K/V 同时写入 KV Cache，供后续解码读取。输入长度增加时，矩阵计算、临时工作集和 KV 写入都会增长。

| 阶段 | 主要动作 | 主要成本 | 直接证据 |
|:---|:---|:---|:---|
| Q/K/V 投影 | 为整段输入生成 Q、K、V | 权重读取、矩阵计算 | 投影算子时间、计算利用率 |
| 分数与 Softmax | 计算词元间关系并逐行归一化 | 中间工作集、HBM 读写 | 注意力算子时间、显存带宽、峰值显存 |
| V 聚合与输出投影 | 汇总上下文并形成层输出 | 矩阵计算、片上复用 | 算子时间、输出一致性 |
| KV Cache 写入 | 保存后续解码需要的 K/V | 缓存容量、写入带宽 | KV 字节数、写入时间、可用容量 |

![预填充与注意力访存](../../docs/public/topic_discussion/inference_optimization/prefill_attention_zh.svg)

### 预填充与解码为何形成不同工作负载

预填充和解码使用相同模型语义，但输入形态不同。预填充通常拥有较大的矩阵和较高并行度；解码每轮只追加少量词元，却持续读取全部历史 KV。预填充路径的收益不能直接外推为 TPOT 收益。

| 阶段 | 输入与状态 | 常见瓶颈 | 优先观察 |
|:---|:---|:---|:---|
| 预填充 | 一次处理完整输入或一个输入分块，并建立 KV 状态 | 注意力工作集、矩阵计算、长输入阻塞 | TTFT、预填充时间、峰值显存、带宽 |
| 解码 | 每轮追加少量词元，并读取历史 KV | KV 读取、批次组织、请求调度 | TPOT、P99、KV 访问量、每轮有效推进量 |

### 三类候选改变什么

FlashAttention、分块预填充（Chunked Prefill）和前缀缓存作用在不同位置。比较时应固定模型、输入、硬件和请求压力，并为每个候选选择能够直接观察其机制的证据。

FlashAttention 的工作集与数值机制见 [Part 02 · 20 FlashAttention / SDPA](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)；分块预填充与阶段调度见 [Part 02 · 38 预填充—解码调度](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb)；前缀匹配与复用见 [Part 02 · 24 Prefix Cache](../../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.ipynb)。

| 现象 | 候选机制 | 最小配对对照 | 必须取得的证据 |
|:---|:---|:---|:---|
| 注意力中间工作集和 HBM 读写过大 | FlashAttention / SDPA 高效路径 | 显式分数矩阵基线 → SDPA 候选 | 输出一致性、算子命中、延迟和峰值显存 |
| 长输入独占一次迭代并阻塞其他请求 | 分块预填充 | 完整预填充 → 固定词元预算的分块预填充 | TTFT、TPOT、P99、迭代组成和解码抖动 |
| 多个请求共享相同输入前缀 | 前缀缓存 | 不复用前缀 → 命中相同前缀 | 复用词元数、命中率、TTFT 和缓存占用 |
| TTFT 高但预填充计算占比不高 | 队列、批次组织或执行等待 | 相同请求下补充队列和迭代事件 | 排队时间、批次成员、GPU 空闲区间 |

### 如何区分计算、访存和启动开销

预填充时间可以拆成矩阵计算、数据搬运以及计算内核启动和同步。Roofline 提供计算上限与带宽上限的判断假设，最终仍需结合算子时间、显存带宽、利用率和端到端 TTFT 验证。

| 成本来源 | 常见信号 | 优先检查 | 可能的动作 |
|:---|:---|:---|:---|
| 计算受限 | 算术强度较高，计算单元利用率高 | FLOPs、Tensor Core 路径、矩阵形状 | 调整数据类型、形状或矩阵计算路径 |
| 访存受限 | 带宽接近上限，计算单元未充分利用 | HBM 流量、工作集、读写次数 | 分块、片上复用、FlashAttention |
| 启动或同步受限 | 单个计算内核很短，但内核数量多且间隙明显 | 启动次数、空闲区间、同步点 | 融合、批处理、CUDA Graph 或编译优化 |
| 形状受限 | 长度变化导致路径和性能抖动 | 填充比例、长度分桶、动态形状 | 分桶、稳定形状、重新组织工作负载 |

### SDPA 如何选择实际计算内核

PyTorch 的 `scaled_dot_product_attention`（SDPA）表达缩放点积注意力语义，运行时再根据数据类型、形状、掩码、设备和后端能力选择实现。调用 SDPA 不等于已经命中 FlashAttention；必须通过性能分析器（profiler）、执行日志或可复查配置确认实际路径。

FlashAttention 后续版本继续改进并行划分、异步流水和低精度硬件路径，但仍需遵守同一验证顺序：先检查输出语义，再确认实际计算内核，最后观察请求指标。[执行分派](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb)是可复用于多类算子的通用机制，本阶段只使用其中与注意力融合、能力检查和即时执行（`eager`）回退有关的路径。

| 层次 | 作用 | 需要确认的证据 |
|:---|:---|:---|
| 注意力 API | 表达输入、掩码和缩放语义 | 输入形状、数据类型、掩码和输出一致性 |
| SDPA 执行分派 | 根据能力条件选择候选或回退 | 实际执行后端、回退原因和版本 |
| FlashAttention 计算内核 | 通过分块和在线 Softmax 减少中间写回 | 内核名称、内核时间、工作集和带宽 |
| 服务执行后端 | 将注意力放入预填充、解码和请求执行 | TTFT、TPOT、并发、峰值显存和 P99 |

## 选择条件与验证证据

先用阶段时间判断问题是否位于预填充，再用工作集、算子路径和请求事件区分注意力计算、长输入阻塞与共享前缀。前文“三类候选改变什么”表中的最小配对对照和证据字段应同时满足；仅有 API 名称或一次 TTFT 变化不能证明目标机制生效。

## 相关阅读

- [FlashAttention：Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)
- [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
