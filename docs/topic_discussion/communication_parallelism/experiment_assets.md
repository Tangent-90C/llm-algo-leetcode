# 通信与并行实验资产表

本表登记通信实验使用的 Profile、比较对象、结果路径和证据状态。环境创建命令见[实验 Quickstart](./quickstart.md)，精确依赖和硬件快照见[GPU 环境资产表](../../gpu_environment_assets.md)。

## 环境角色

| 环境 ID | 直接入口 | 可形成的证据 |
|:---|:---|:---|
| `PAR-CPU` | 所有机制页 Step 1–4 | 切分、状态、路由、账本和决策机制 |
| `PAR-CUDA-MECHANISM` | 27–29、46–48、PAR-CONTEXT、PAR-REPLICA | 单机多卡 collective、rank 布局、payload、局部计算与暴露通信 |
| `PAR-TRAIN-CLUSTER-RAY`（规划） | 79、PAR-MODEL-INTERNAL、80 的跨节点项目 | Ray Train 管理 worker、资源预留、失败恢复和任务编排；底层张量通信仍由 PyTorch Distributed / NCCL 完成 |
| `PAR-SERVING-RAY`（规划） | 81 的集群 Serving 路径 | Ray Serve 管理副本、placement group、跨节点 worker 和弹性伸缩；模型执行与缓存指标仍由 vLLM / SGLang 产生 |
| 独立 Serving backend | 81 的单机或不使用 Ray 的路径 | vLLM / SGLang 的请求、模型并行、副本和服务指标 |

`PAR-CUDA-MECHANISM` 不安装 Ray。机制页需要直接观察 rank、collective、payload 和暴露通信时间；引入 actor 调度、对象存储和资源编排会增加与通信机制无关的变量。Ray 只在项目需要跨节点资源编排、gang scheduling、故障恢复、服务副本或弹性伸缩时进入独立环境，也不能把 placement group 创建成功写成 NCCL 或并行效率证据。

## 机制实验注册表

| 实验 ID / 页面 | 固定对照 | 主要指标 | 当前状态 |
|:---|:---|:---|:---|
| `PAR-ZERO-STATE` / 27 | DDP → ZeRO-1/2/3 | 状态字节、峰值显存、通信量、step time | 5.1–5.4 已统一；真实多卡待验证 |
| `PAR-PIPELINE-SCHEDULE` / 28 | 单 stage → 两 stage；fill-drain → 1F1B | bubble、activation 峰值、stage time、吞吐 | 5.1–5.4 已统一；真实多卡待验证 |
| `PAR-TENSOR-LAYOUT` / 29 | 单卡 MLP → Column/Row TP | 数值等价、collective bytes、延迟、显存 | 5.1–5.4 已统一；真实多卡待验证 |
| `PAR-EXPERT-DISPATCH` / 46 | balanced → skewed routing | capacity、overflow、All-to-All bytes、max/mean load | 页面与 GPU 契约已统一；真实多卡待验证 |
| `PAR-NCCL-COLLECTIVE` / 47 | AllReduce / AllGather / ReduceScatter × payload | median/P95、有效带宽、每 rank 时间 | 公共多卡基线已统一；真实多卡待验证 |
| `PAR-OVERLAP-HOTSPOT` / 48 | serial → overlap | 通信总时长、暴露通信、overlap ratio、rank imbalance | 已复用公共身份与结果字段；真实多卡待验证 |
| `PAR-CONTEXT-RING` / PAR-CONTEXT | 复制完整 K/V → 序列切分与交换 | 正确性、状态字节、交换字节、延迟 | 机制实验已统一；Ring kernel 证据待项目验证 |
| `PAR-REPLICA-ROUTING` / PAR-REPLICA | local / balanced / locality-aware | makespan、每副本负载、队列等待 | 只形成副本机制证据 |

49 是 CPU 策略决策页，消费上述资产，不单独制造 GPU 结果。

## 项目接口

项目页复用机制字段，但必须重新固定真实模型、输入、拓扑与执行栈；机制 JSON 不能直接充当项目结论。

| 稳定项目 ID | 消费的机制资产 | 项目补充的证据 | 当前接口状态 |
|:---|:---|:---|:---|
| `PAR-TRAINING-STATE` / 79 | 27、46、48 | DDP / FSDP / ZeRO 的真实训练显存、吞吐与通信 | 已接 `distributed-benchmark/v1` 聚合；真实多卡待验证 |
| `PAR-MODEL-INTERNAL` | 28、29、PAR-CONTEXT、46、48 | PP / TP / CP 与混合布局的真实模型结果 | CPU 项目已建设；真实执行 adapter 待接入 |
| `PAR-MOE-EXPERT` / 80 | 47、46、48 | 真实 MoE workload 的有效 token、overflow、All-to-All 与长尾 | 已接聚合器；真实 MoE backend 待接入 |
| `PAR-DISTRIBUTED-INFERENCE` / 81 | PAR-REPLICA、29、PAR-CONTEXT、48 | 副本、模型并行与 PD 分离的 Serving 指标 | 5.1–5.6 已分层；vLLM / SGLang adapter 待接入 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、页面与策略 |
| 硬件与拓扑 | GPU、world size、rank-device 映射、单机/跨机、链路信息 |
| 通信 | backend、collective、payload bytes、调用次数、同步频率 |
| 时间 | compute、communication、exposed communication、median/P95 |
| 负载 | 每 rank 时间或工作量、max/mean、overflow 或空闲比例 |
| 结论 | correctness、failure、fallback、`evidence_level`、decision |

机制实验不能把环境预检写成通信证据，也不能把单机双卡结果外推为跨机扩展效率。项目页必须固定真实模型、输入和执行后端后重新测量。
