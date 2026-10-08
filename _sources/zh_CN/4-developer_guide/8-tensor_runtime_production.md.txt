# Tensor Runtime 生产化

本页定义 scoped 单 GPU tensor runtime 的 review 所有权与验收规则。操作复现
说明见
{doc}`../2-user_guide/1-training/7-tensor_runtime_production`；生成的支持声明
见 {doc}`../5-reference/5-support_matrix`。

## 契约所有权

| Owner | 职责 |
| --- | --- |
| UniSim | 公共 `SimBackend` tensor lifecycle、execution/data-plane/process profile、packed host-bridge 契约、平台支持 inventory 与 backend adapter。 |
| UniLab task owner | Hydra owner 身份、`training.sim_backend`、task 语义、parity fixture、进程设备绑定与训练入口集成。 |
| `uni_rl` | off-policy collector/learner 进程拓扑、CUDA inference ring、replay ingress、IPC 同步、runner metric 与 metric/runtime-manifest producer schema。 |
| UniLab training runtime | soak 监控、schema 消费、安全默认值与边界、process-affinity 集成、run artifact 与 fail-closed 构造检查。 |
| UniLab 文档 | 复现、运维、schema 兼容、support-matrix 解读与 release-transition 指南。 |

Task 与 training 代码只能消费公共 `SimBackend` 接口，不得探测 backend 私有
方法，也不得把 `unknown` 平台能力提升为支持。不支持的 tensor lifecycle 必须
fail closed。

## Schema 与版本策略

当前 namespace 相互独立：

| Namespace | 版本 | Producer owner |
| --- | ---: | --- |
| TensorBoard / W&B metric schema | `1` | `uni_rl.logging.metric_schema` |
| Runtime 进程 / 设备 / IPC manifest | `1` | `uni_rl.logging.runtime_manifest_schema` |
| UniLab 长 soak 监控 artifact | `0.3.0` | `unilab.training.soak` |

同一 major version 内，producer 可以伴随 schema、测试与文档增加可选字段。
稳定字段不能改名、删除、改变类型，或改变单位与 null 语义。Consumer 遇到
缺失或不支持的版本必须拒绝猜测。soak artifact 升级不改变 metric 语义，
metric schema 升级也不会让所有 runtime-manifest 字段失效。

完整兼容契约维护在
{doc}`../2-user_guide/1-training/3-logging`。不要把该表复制到 backend 或
task 文档中形成第二事实源。

## Benchmark 验收

有两类不可互换的证据：

1. canonical G1 FlashSAC / MJWarp 长 soak 是 owner-routed 端到端训练
   lifecycle 测试。当前 head 通过要求当前 schema、训练完成、inference 与
   replay ingress 排空、normal shutdown、无 residual process 且资源增长有界。
2. phase-local backend probe 测量同步后的 backend/manager phase，并排除
   inference IPC、replay ingestion、learner update 与生产 dispatch。其结果是
   诊断证据，不能报告为端到端训练吞吐。

历史 soak artifact 只对它记录的 commit 与 schema 保持 lifecycle 证据价值。
producer schema 或 lifecycle 契约变化后，它不能替代当前 head 运行。

保留 benchmark 证据时，应同时保存 soak JSON、console log、run 目录、
revision 与包来源、GPU 身份与 checksum。仓库不跟踪机器生成的 benchmark
输出。

## Support-Matrix 提升

生成的 matrix 是权威。本地 benchmark 通过不会自动改变 entrypoint 证据等级。
提升为 `Tested` 或 `Benchmarked` 需要窄范围的 maintainer-validation 记录或
checked-in benchmark manifest、generator 更新、focused tests，以及重新生成的
中英文 block。

当前 FlashSAC / `g1_motion_tracking` / MJWarp 生产指南仍是 `Configured` owner
路径。它不是默认推荐，也不做 multi-GPU、macOS、ROCm、PyPI 发布或通用
backend 声明。

## Release Transition

推进已发布依赖版本前，必须完成 {doc}`5-contributing_workflow` 中的 checklist：已发布
的 UniSim tensor 契约与 adapter、已发布的 `uni_rl` runner 行为、已发布且受支持的
MJBatch、在确切 transition commit 上通过的 focused CPU 与 gated CUDA profile，
以及完整的 M11 soak、泄漏、shutdown、schema、平台与文档验收。

M11 不包含任何 PyPI 发布。发布需要单独的 maintainer 决策与 release tag。
