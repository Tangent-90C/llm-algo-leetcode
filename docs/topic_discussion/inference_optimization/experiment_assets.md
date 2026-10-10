# 推理优化实验资产表

本表登记推理机制实验与项目出口。环境创建见[实验 Quickstart](./quickstart.md)，统一字段见[Part 02 结果资产与证据契约](../../part02_result_assets.md)。

## 环境角色

| 环境 ID | 直接入口 | 可形成的证据 |
|:---|:---|:---|
| `INF-CPU` | 请求、预算、缓存、调度与投机解码的 Step1–4 | 状态、公式、策略与决策逻辑 |
| `INF-CUDA-MECHANISM` | 20、迭代预算、执行分派、动态候选预算与状态重叠 | 真实模型或真实 CUDA 路径的正确性、延迟、显存、能力门槛与回退证据 |
| `INF-VLLM-011-CU128` | 66–72 的主服务路径 | 固定请求清单下的 TTFT、TPOT、吞吐、P95/P99 与峰值显存 |
| SGLang 独立环境 | 69、70、72 的可选同口径对照 | Prefix/Radix Cache、调度和模型兼容性证据 |
| 通信与并行 Profile | 81 与模型内部并行路径 | 多卡通信、扩展效率和分布式失败证据 |

机制环境不安装服务执行后端。服务能够启动只形成环境或能力预检；稳定性能结论必须来自固定模型版本、请求清单、并发和重复次数的配对实验。

新增机制页分为两层：请求生命周期、迭代预算、执行分派、EAGLE、动态候选预算、MLA、连续批处理和执行重叠属于核心或共享机制；DFlash / DSpark 与 AFD 属于扩展分支。扩展页不要求首次学习，也不为了页面齐全重复建设低价值 GPU 实验。

## 机制实验注册表

| 机制与页面 | GPU 角色 | 固定对照与指标 | 当前状态 |
|:---|:---|:---|:---|
| 请求生命周期 / `INF-REQUEST-LIFECYCLE` | 不单设 GPU | 事件轨迹、TTFT、TPOT、E2E、失败状态 | CPU 机制完整；66 提供真实后端基线 |
| Attention 与执行 / 20、`INF-EXECUTION-DISPATCH` | 单卡机制 | 显式 Attention / SDPA、eager / compile 候选；输出一致性、延迟、峰值、路径与回退 | 两页均已接入 `INF-CUDA-MECHANISM`；待真实单卡复测 |
| 迭代预算 / `INF-ITERATION-BUDGET` | 单卡机制 | 批量 × 上下文；中位数/P95、KV 字节、峰值、OOM 恢复与拐点 | 已统一 Profile 与结果契约；待真实单卡复测 |
| 连续批处理 / `INF-CONTINUOUS-BATCHING` | 项目承接 | static / continuous；occupancy、queue wait、TTFT、TPOT、P99 | CPU 机制完整；70 提供服务证据 |
| 动态候选预算 / `INF-PROPOSAL` | 单卡机制 | Transformers 辅助生成的固定、启发式与动态调度；输出一致性、目标前向次数、中位数/P95 与净收益 | 已接入 `INF-CUDA-MECHANISM`；待真实单卡复测，结果再交给 68 扩大到专项后端 |
| 草稿架构 / 35、EAGLE、并行草稿 | 项目承接 | MTP / EAGLE / DFlash / DSpark；接受率、有效推进、质量与净收益 | EAGLE 为核心候选，DFlash / DSpark 为扩展；配对权重与 Speculators/vLLM 证据由 68 承接 |
| KV 生命周期与复用 / 22、24、34、37 | 执行后端机制 + 项目承接 | 容量、命中、引用、驱逐、回收和 TTFT | 37 已建立唯一/共享前缀配对契约；22 不再保留合成显存探针，69 提供稳定服务证据 |
| 请求与阶段调度 / 36、38、39 | 项目承接 | 排队时间、TTFT、TPOT、P95/P99、交接与暴露时间 | 机制页不保留随机矩阵或环境预检作为 GPU 实验；70/81 提供真实服务证据 |
| 执行重叠 / `INF-SERVING-OVERLAP` | 单卡 CUDA 机制 + 项目承接 | 串行 / CUDA Stream；正确性、传输、暴露时间、中位数/P95 | 已接入 PyTorch CUDA Stream 配对矩阵；队列与状态交接由 70 或分阶段项目承接 |
| AFD / `INF-AFD` | 可选多 GPU / 执行后端 | 普通 MoE / AFD；正确性、通信、暴露时间、TTFT、TPOT 与吞吐 | 5.1–5.4 已统一；待固定插件、模型、适配器和真实多 GPU 环境 |

## 项目出口

| 项目 ID | 页面 | 核心职责 |
|:---|:---|:---|
| `INF-FLOAT-BASELINE` | 66 | 建立浮点模型和统一请求指标基线 |
| `INF-SPECULATIVE-BENCH` | 68 | 比较草稿拓扑、验证语义、接受率与净收益 |
| `INF-PREFIX-CACHE` | 69 | 验证前缀复用、缓存命中和 TTFT 收益 |
| `INF-SERVING-SCHEDULER` | 70 | 比较请求选择、连续批处理、阶段调度与 overlap |
| `INF-SINGLE-GPU-DEPLOY` | 72 | 单卡模型、后端、容量和部署决策 |
| `DIST-INFERENCE` | 81 | 单实例不足后的多卡执行与扩展效率 |

项目只消费机制页定义的状态和指标，不重新定义机制。跨专题只交换版本化 JSON 或产物清单，不依赖另一 Notebook 的内存变量。

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、模型和分词器版本 |
| 工作负载 | 输入/输出长度、到达模式、并发、批量/Token 预算、随机种子 |
| 环境 | Profile ID、Python、PyTorch、CUDA、GPU、执行后端与版本 |
| 性能 | TTFT、TPOT、E2E、throughput、P50/P95/P99、queue wait、peak memory |
| 正确性 | 输出数量、终止状态、质量或与参考输出的一致性 |
| 结论 | `failure`、`evidence_level`、`decision` 与下一步 |
