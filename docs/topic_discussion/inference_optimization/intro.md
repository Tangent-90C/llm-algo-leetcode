# 推理优化（Inference Optimization）

> 专题类型：主学习路线　主服务目标：请求性能与 Serving 决策

推理优化研究模型如何把一组输入 Token 转换为连续输出，以及请求在排队、预填充（Prefill）、解码（Decode）、KV Cache 和执行后端中为何产生延迟、容量与吞吐差异。学习重点是把请求现象还原为状态、工作负载、执行路径和可比较证据。

本路线先固定请求生命周期和指标口径，再进入 Attention 执行、投机式生成、KV Cache、服务调度、压缩部署和分布式推理。机制页解释单个状态变化或执行选择，项目页在固定工作负载中比较真实后端并形成部署决策。

## 学习路线

按 Task 顺序学习时使用下表；需要连续理解请求因果关系时进入[推理优化问题链](./walkthrough.md)，遇到延迟、容量、缓存或回退异常时使用[推理优化判断手册](./casebook.md)。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 请求、工作负载与指标 | 一次请求如何经过队列、预填充、解码和终止，指标如何对应这些阶段？ | 核心：[Part 02 · INF-REQUEST 请求生命周期](../../02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md)；项目：[Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) | 请求状态 → 阶段指标 → 工作负载与结果契约 | [专题正文 · 01 请求链路与指标](./01_request_path_and_metrics.md) |
| Task1 · Attention 计算与执行分派 | Attention 的表示、工作集、融合和执行后端如何改变预填充与解码成本？ | 核心：[Part 01 · 04 Attention 显存优化](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md) → [Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md) → [Part 02 · 04 MHA / GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) → [Part 02 · 20 FlashAttention / SDPA](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md) → [Part 02 · INF-EXECUTION-DISPATCH](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)；项目：[Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)；扩展：[算子优化](../operator_optimization/intro.md) | Head 表示 → 工作集与融合 → 执行后端能力门槛 → 预填充 / 解码证据 | [专题正文 · 02 预填充与 Attention 计算内核](./02_prefill_and_attention_kernel.md) |
| Task2 · 解码推进与候选验证 | 草稿、验证和动态预算如何在保持目标分布的前提下提高有效推进量？ | 核心：[Part 02 · 21 解码策略](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md) → [Part 02 · 23 投机解码](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md)；草稿机制选择 [Part 02 · 35 原生 MTP](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md)或[Part 02 · INF-EAGLE](../../02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md)，再进入[Part 02 · INF-PROPOSAL 动态预算](../../02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md)；项目：[Part 02 · 68 投机解码基准](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md)；扩展：[Part 02 · INF-PARALLEL-DRAFTING](../../02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md) | 普通解码 → 验证语义 → MTP / EAGLE 草稿分支 → 动态预算 → 净收益 | [专题正文 · 03 解码策略](./03_decoding_strategies.md) |
| Task3 · KV Cache 表示、管理与复用 | KV 如何表示、分页、追加、复用、驱逐与压缩？ | 核心：[Part 01 · 11 KV Cache 与显存增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) → [Part 02 · ARCH-MLA](../../02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.md) → [Part 02 · 22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) → [Part 02 · 37 KV 生命周期](../../02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.md) → [Part 02 · 24 Prefix Cache](../../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.md) → [Part 02 · 34 RadixAttention](../../02_PyTorch_Algorithms/34_SGLang_RadixAttention.md)；项目：[Part 02 · 69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)、[Part 02 · 71 MLA](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md)；扩展：[Part 02 · 54 KV Cache 量化](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md) | 容量增长 → 紧凑表示 → 分页与生命周期 → 前缀复用 → 压缩 | [专题正文 · 04 KV Cache 表示、管理与复用](./04_kv_cache_lifecycle_and_reuse.md) |
| Task4 · 服务迭代预算、调度与异构协同 | 单轮工作量上界如何进入请求选择、动态批次、阶段调度与异构路由？ | 核心：[Part 02 · INF-ITERATION-BUDGET](../../02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) → [Part 02 · 36 请求调度](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md) → [Part 02 · INF-CONTINUOUS-BATCHING](../../02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching.md) → [Part 02 · 38 阶段调度](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.md) → [Part 02 · INF-SERVING-OVERLAP](../../02_PyTorch_Algorithms/INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md) → [Part 02 · 39 异构 PD](../../02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.md)；项目：[Part 02 · 70 Serving 调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)；扩展：[Part 02 · INF-AFD](../../02_PyTorch_Algorithms/INF-AFD_Attention_FFN_Disaggregation.md) | 读取 Task3 容量状态 → 单轮预算 → 请求选择 → 动态批次 → 阶段调度 → 重叠与异构路由 | [专题正文 · 07 Serving 调度与 PD 分离](./07_serving_scheduling_and_pd.md) |
| Task5 · 量化推理与单卡部署验证 | 量化后的权重和运行配置如何被加载器与目标执行后端正确使用，并在单卡上产生可复现收益？ | 核心：[Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md)建立表示基线；按量化对象选择[Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md)、[Part 02 · 53 SmoothQuant](../../02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant.md)或[Part 02 · 41 FP8 运行时](../../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md)，再用[量化与压缩 · Task5](../quantization_compression/intro.md)检查量化文件、加载器、计算内核和回退路径；项目：[Part 02 · 82 量化产物准入](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.md) → [Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)，以及[Part 02 · 66 浮点基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) → [Part 02 · 72 单卡部署](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.md)；扩展：[Part 02 · 83 量化 Serving Profile](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.md) | 浮点基线 → 权重 / 激活 / 运行时量化分支 → 加载与回退检查 → 单卡配对复测 | [专题正文 · 05 量化推理与部署](./05_quantized_inference_and_deployment.md) |
| Task6 · 分布式扩展与端到端决策 | 单卡基线不足时，副本或模型内部并行能否在通信代价下改善目标指标？ | 核心：[专题正文 · 08 分布式推理与并行](./08_distributed_inference_and_parallelism.md)；[Part 01 · 26 并行策略决策](../../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md) → [Part 02 · PAR-REPLICA](../../02_PyTorch_Algorithms/PAR-REPLICA_Serving_Replica_Routing.md)，或[Part 02 · 28 Pipeline Parallel](../../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md) / [Part 02 · 29 Tensor Parallel](../../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md) / [Part 02 · PAR-CONTEXT](../../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md) → [Part 02 · 49 策略选型](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)；项目：[Part 02 · 87 模型内部并行](../../02_PyTorch_Algorithms/87_Model_Internal_Parallel_Benchmark.md) → [Part 02 · 81 分布式推理验证](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md)；扩展：[通信与并行](../communication_parallelism/intro.md) | Task5 单卡基线 → 副本 / 模型内部并行 → 通信代价 → 扩展效率 → 统一决策 | [专题正文 · 06 端到端基准与决策](./06_benchmark_and_decision.md) |

![推理优化学习路线：从请求链路到专项验证](../../public/topic_discussion/inference_optimization/inference_optimization_roadmap.svg)

![推理优化知识地图：从请求对象到验证决策](../../public/topic_discussion/inference_optimization/inference_optimization_knowledge_map.svg)

## 扩展与跨专题入口

### 按需回补：架构与共享前置

遇到架构、MoE、dtype 或精度方面的知识缺口时，从下表回补对应入口，再回到当前 Task 继续学习。

| 补充主题 | Notebook 入口 | 建议时机 |
|:---|:---|:---|
| 模型计算量与硬件资源 | [Part 01 · 02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md) → [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) | Task0–1；需要估算计算、带宽与容量上界时 |
| 架构基础 | [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)、[Part 02 · 08 Architecture Tricks](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) | Task0 后需要补模型数据流时 |
| 数据类型、量化表示与运行时约束 | [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) → [Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [Part 01 · 34 异构运行时约束](../../01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.md) | Task5；需要解释数据类型、产物兼容性或执行回退时 |
| 异步执行与状态交接 | [Part 01 · 17 CUDA Stream](../../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md) | Task4；需要解释准备、计算和搬运重叠时 |
| 通信拓扑与集合通信 | [Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md) → [Part 01 · 20 NCCL 与 AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md) | Task6；需要解释多卡通信上界时 |
| MoE 架构与计算 | [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md)、[Part 02 · 07 MoE Load Balancing](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md)、[Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md) | Task4 / Task6；进入 AFD 或专家并行前 |

### 跨专题入口

需要定位延迟和容量来源时进入[性能分析](../performance_analysis/intro.md)，需要比较候选和瓶颈转移时进入[性能优化](../performance_optimization/intro.md)，量化产物与低精度路径进入[量化与压缩](../quantization_compression/intro.md)，多卡切分与通信证据进入[通信与并行](../communication_parallelism/intro.md)，计算内核与融合进入[算子优化](../operator_optimization/intro.md)。

## 实验与项目出口

首次运行先按[实验 Quickstart](./quickstart.md)创建或恢复环境，再从[实验资产表](./experiment_assets.md)选择 Profile、模型、工作负载、候选和结果路径；跨专题的软件版本与硬件快照查询[GPU 环境资产表](../../gpu_environment_assets.md)。机制页验证一个受控变量，项目页再进入真实执行后端、请求队列和部署决策。

| 证据阶段 | 要回答的问题 | 输出与入口 |
|:---|:---|:---|
| CPU 机制 | 状态转换、公式、预算或选择规则是否成立？ | 各机制页的题目、测试、答案与解析 |
| 单卡机制 GPU | 真实模型、算子或运行时路径是否改变局部质量、延迟或显存？ | [Part 02 · 20 FlashAttention / SDPA](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)、INF-ITERATION-BUDGET、INF-EXECUTION-DISPATCH、INF-PROPOSAL 与 INF-SERVING-OVERLAP 的机制 JSON |
| 专项后端 | 缓存、调度或投机策略是否在真实请求中成立？ | [Part 02 · 68 投机解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md)、[Part 02 · 69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)、[Part 02 · 70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md) |
| 单卡部署 | 合格产物是否被目标执行后端正确加载并产生可复现收益？ | [Part 02 · 66 浮点基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) → [Part 02 · 72 单卡部署](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.md)；复用 [Part 02 · 67](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) 与 [Part 02 · 71](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md) 的专项证据 |
| 分布式决策 | 多卡方案能否在通信代价下改善容量、延迟或吞吐？ | [Part 02 · 87 模型内部并行](../../02_PyTorch_Algorithms/87_Model_Internal_Parallel_Benchmark.md) → [Part 02 · 81 分布式推理验证](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) |
