# 01. Model Artifact and Deployment Contract | 模型产物与部署契约

## 问题起点

为什么“有权重文件”不等于得到一个可部署候选？部署时实际被执行的是模型、revision、tokenizer、template、artifact 与 runtime 配置的组合。缺失其中任一项，后续 benchmark 即使有数字也无法可靠复现或比较。

## 核心机制一：候选身份与输入契约

![部署契约：从模型来源到可观测结果](../../public/topic_discussion/deployment_heterogeneous/deployment_contract.svg)

| 契约层 | 应固定的内容 | 常见失配 |
|:---|:---|:---|
| 模型身份 | 名称、revision、架构、许可和来源 | 同名模型不同 revision，权重或配置已变更 |
| 输入语义 | tokenizer、chat template、特殊 token、模态处理器 | prompt 看似相同，但实际 token 序列不同 |
| artifact | 权重格式、量化方式、scale 元数据、校验和 | 文件可见但 backend 无法识别或缺少配套文件 |
| runtime | engine、版本、dtype、关键配置和启动参数 | 默认配置改变了上下文、缓存或执行路径 |
| 输出与质量 | 采样参数、停止条件、任务指标和失败规则 | 只比较延迟，没有确认输出是否仍满足目标 |

部署记录的最小单位不是“模型名”，而是 `model + revision + artifact + tokenizer/template + backend config + workload`。这份组合就是后续比较中的候选身份。

## 核心机制二：manifest 与三道检查门

manifest 不替代模型仓库或日志；它把复测所需的关键引用汇总到一次实验旁边。建议每个候选至少写入：

```text
artifact_id / model_revision / tokenizer_revision / format / checksum
backend / backend_version / precision / launch_config
hardware / runtime / workload_id / quality_protocol
```

当加载失败时，也保留同一份 manifest 和错误摘要。失败是兼容性证据，不应被静默丢弃。

### 可复用的 artifact manifest

下面的结构不绑定某个 backend。项目节可以增加各自的专属字段，但不应省略候选身份、输入契约和质量协议。

```json
{
  "candidate_id": "qwen-text-int4-v1",
  "model": {"name": "", "revision": "", "architecture": ""},
  "input_contract": {"tokenizer_revision": "", "chat_template": "", "modalities": ["text"]},
  "artifact": {"format": "", "precision": "", "path": "", "checksum": ""},
  "quality_gate": {"dataset": "", "metric": "", "threshold": null},
  "provenance": {"source": "", "created_at": "", "notes": ""}
}
```

`candidate_id` 用于把 artifact、启动配置和结果 JSON 串起来；它不是对模型能力的命名。若只修改量化格式、revision 或 template 中任一项，就应生成新的候选身份，避免把不可比较的结果混在一起。

### 三道检查门

| 检查门 | 必须确认的事实 | 失败后的正确处理 |
|:---|:---|:---|
| 身份门 | 模型、revision、tokenizer、template 与 artifact 引用完整 | 补 manifest；不要开始性能比较 |
| 加载门 | 目标 runtime 能识别 artifact 并完成最小请求 | 记录兼容性失败和环境，不填写性能字段 |
| 质量门 | 固定质量集上的输出满足阈值或人工复核规则 | 标记为 `reject` 或回到量化/训练调整，不用速度收益覆盖质量问题 |

这三道门的顺序很重要：性能 benchmark 是加载门和质量门之后的工作。`82` 适合承载 artifact 与质量 gate；`72`、`83` 承接通过 gate 后的服务比较。

![部署候选的身份、加载与质量检查门](../../public/topic_discussion/deployment_heterogeneous/deployment_contract_gates.svg)

## 证据与判断

一个 artifact 通过加载门，只说明当前环境可以启动；通过质量门，才具备进入性能比较的资格。任何缺失 revision、template 或质量协议的结果，都应标为“候选身份不完整”，而不是与其他候选比较速度。

## 项目出口与相关阅读

量化专题负责解释 format、scale 与误差；后训练专题负责产生 adapter、merged model 或 checkpoint。部署专题检查这些产物是否携带完整身份、能否被目标 runtime 加载，并在同一质量协议下进入比较。

下一步阅读 [02 Backend 兼容性与单机执行](./02_backend_compatibility_and_single_node_execution.md)，把静态 artifact 接到实际 runtime。
