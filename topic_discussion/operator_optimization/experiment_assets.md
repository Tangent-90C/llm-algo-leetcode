# 算子优化实验资产表

本表登记算子优化需要的证据层、可复用环境和当前缺口。Part 03 已有 `llm_algo_cnb_gpu` 环境，由 `cnb/environment-gpu.yml` 组合 `requirements/gpu.txt` 安装 PyTorch、CUDA 运行时与 Triton；创建、更新和预检步骤见[算子优化 Quickstart](./quickstart.md)。它是 Chapter 3 的既有 GPU 环境，不冒充尚未注册的统一 Profile。

## 环境与证据分工

| 环境或证据层 | 直接入口 | 可以回答的问题 | 不能替代 |
|:---|:---|:---|:---|
| CPU 参考实现 | Part 01 机制练习、Part 03 正确性单元 | 形状、边界、掩码、公式和数值容差 | GPU 计算内核、带宽、占用率和启动成本 |
| `PERF-CUDA-MECHANISM` | [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb)、[Part 02 · 20 FlashAttention 模拟](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) | PyTorch 工作负载的资源上界、执行路径和配对性能 | Triton 或自定义 CUDA 计算内核已正确编译并命中 |
| `llm_algo_cnb_gpu` | [Part 03 · 01–13](../../03_Triton_Kernels/intro.md) | Triton 正确性、微基准、自动调优和局部组合 | CUDA 专用路径、其他硬件或模型端到端收益 |
| Part 04 CUDA / 系统环境（待固定） | [Part 04 · 21 CUDA / Triton / PyTorch](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.ipynb) | 实现路径、CUDA 下探和运行时差异 | 其他设备、形状或生产负载的普遍结论 |

## 验证顺序

| 顺序 | 验证对象 | 必须固定或保存 | 进入下一步的条件 |
|:---|:---|:---|:---|
| 0 | 语义与边界 | 参考实现、形状、布局、数据类型、容差和失败输入 | 边界形状与随机输入全部通过 |
| 1 | 目标执行路径 | 编译目标、后端、计算内核名称和回退原因 | 证据证明候选实际命中目标路径 |
| 2 | GPU 微基准 | 设备、预热、重复样本、同步方式和编译缓存 | 候选在目标形状分布上有稳定收益 |
| 3 | 资源归因 | FLOPs、字节、寄存器、共享内存、占用率与执行轨迹 | 观察结果支持原瓶颈假设，或明确推翻它 |
| 4 | 模型工作负载复测 | 固定模型、输入、质量、局部与端到端样本 | 局部收益能够传递，且守护条件通过 |
| 5 | 采用判断 | 失败范围、可移植性、编译和维护成本 | 形成 `accept / tune / reject` 结论 |

## 当前实验注册表

| 实验或页面 | 固定对照 | 结果资产 | 当前状态 |
|:---|:---|:---|:---|
| [Part 02 · 20 FlashAttention 模拟](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) | 数学等价的注意力路径与固定序列矩阵 | `benchmarks/results/20_flashattention/<semantic_id>_<run_label>.json` | RTX 5070 Ti 已有 SDPA 路径结果；只能证明当前 PyTorch 后端对照 |
| [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) | 实测带宽、计算上界与候选工作量 | `benchmarks/results/perf_roofline/<workload_id>_<run_label>.json` | 已有单卡结果，可作为候选约束证据 |
| [Part 03 · Triton 计算内核](../../03_Triton_Kernels/intro.md) | PyTorch 参考实现与 Triton 候选 | 待统一登记 | GPU 环境已有；统一结果模式与集中 GPU 复测尚未完成 |
| [Part 03 · 13 LLaMA3 Block](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb) | 参考 Block 与候选计算内核组合 | 待统一登记 | 项目出口存在；需在 Part 03 稳定后进行端到端验收 |

没有目标计算内核的路径证据时，配置开关不能证明优化已经生效；只有微基准结果时，结论不得扩展为模型或服务端到端收益。
