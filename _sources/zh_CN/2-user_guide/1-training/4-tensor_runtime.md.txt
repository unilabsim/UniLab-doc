# Tensor runtime

SAC、FlashSAC 与 WarpSAC 会在探测环境或构造 learner 之前解析所有面向进程
启动的 tensor-runtime 设置。owner YAML 中的值会被收敛为一个有边界的
`TensorRuntimeSettings` 对象并传给 off-policy runner。因此，非法值、超出
边界的数据以及显存预算不可能满足的组合，都会在 collector 或 learner 进程
启动前失败。

该契约只覆盖当前 scoped 单 GPU off-policy runtime；它不是自动调参器，也不
提供多 GPU scaling。

## Owner 默认值与边界

| 设置 | 默认值 | 有效范围或约束 |
| --- | ---: | --- |
| `training.inference_slot_capacity` | `1` | 正整数，最大 `16` |
| `training.collector_metrics_interval` | `1` | 正整数，最大 `10000` |
| `training.replay_ingress_depth` | `2` | 正整数，最大 `16` |
| `training.replay_ingress_slot_rows` | `null` | `1` 到 `algo.num_envs`；`null` 表示 `algo.num_envs` |
| `training.cuda_process_sharing` | `null` | `null` 或单 rank MJWarp 显式 `mps` |
| Learner 每次同步的行数 | `algo.batch_size * algo.updates_per_step` | 受 CUDA 显存预算约束 |

除 `training.cuda_process_sharing` 外，所有值必须是精确的正整数。布尔值、字符串、
浮点数、零和负数都会 fail closed。`training.cuda_process_sharing` 只接受 `null`
或显式字符串 `mps`。G1 Motion Tracking / MJWarp owner 仅有意将
`collector_metrics_interval` 覆盖为 `100`；其他 tensor-runtime 默认值仍保持
上表所示。

## Replay ingress 权衡

默认 `replay_ingress_slot_rows: null` 会解析为 `algo.num_envs`，并为每个
collector vector 发布一个 ingress chunk。这使 publication 与同步模式保持
可预测，是推荐的生产默认值。

将该字段设置为更小的值可以把一个 collector vector 切成连续 chunk，从而降低
常驻 ingress 内存。它只应用于显式的内存压力调参。chunk 数量增加意味着更多
publication、更多 producer backpressure 机会，并且在 CUDA 上每个发布的 chunk
都会引入一次 current-stream barrier。采用更小的值前，务必在目标 workload 上
重新 benchmark。

Ingress diagnostics 统计的是 publication chunk，而不是完整的逻辑 collector
vector。shutdown 时可能保留一个尚未完整发布 vector 的有效行前缀，但不会发布
非法 transition 行。replay ingress 计数器与 shutdown schema 见
{doc}`3-logging`。

## 随机数生成

通用 Manager seam 仍由 NumPy 拥有：`ManagerBasedRlEnv.rng` 是
`np.random.Generator`，既有种子流不变。已验证的 device-resident owner 可以
改为持有 `unilab.managers.TorchManagerRng`。该适配器包装一个显式 CPU/CUDA
`torch.Generator`，校验种子和未支持的采样语义，并始终在该 generator 的设备
上返回 tensor。它不是 NumPy bitstream 兼容层。

当前 scoped G1 Motion Tracking / FlashSAC owner 使用 CUDA
`TorchManagerRng` 驱动观测噪声、adaptive/mixed motion sampling 与
selected-row reset。因此 A/B 评估比较同一整数种子下的可复现性和分布，而不是
逐位比较 NumPy 轨迹。确定性运行必须显式设置 `env.seed`；`algo.seed` 只 seed
训练进程的全局 Python/NumPy/Torch RNG，不会替换 owner 本地 generator 的种子。

## Runtime evidence

Runtime manifest 记录审计 run 所需的有效证据：

- `runtime_limits` 包含已解析 tensor-runtime knob 的 configured、default、
  effective 与 maximum 值。
- `inference_memory_budget` 记录有边界的 CUDA inference-ring 预算。
- `tensor_memory_budget` 记录 CUDA inference、replay storage、replay ingress、
  learner batch 与 workspace 的组合预算计算。
- owner 显式请求 `mps` 时，`cuda_process_sharing` 记录已验证的
  execution-sharing 证据；默认 `null` 模式下该字段不存在。

预算决策在 spawn 前写入 manifest。如果不安全的组合被拒绝，错误会指出具体
设置，且相关进程不会被启动。

单 GPU G1 FlashSAC/MJWarp 生产工作流的 benchmark 复现、artifact 与故障排查见
{doc}`7-tensor_runtime_production`。
