# 运行时模型

详细的运行时契约见
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` 与
{doc}`/zh_CN/4-developer_guide/0-index`。本页将运行时摘要与对应的代码路径放在
一起说明。

## 两种运行时形态

### 同步 PPO 路径

`src/unilab/scripts/train_rsl_rl.py` 会 compose Hydra config、
调用 registry bootstrap、通过 `registry.make(...)` 构造严格 `TorchEnv`，并在同一
进程内运行 learner。默认配置保持单进程；`CUDA_VISIBLE_DEVICES` 指定多张卡时，父进程
通过 PyTorch elastic launcher 启动本机 worker，worker 再进入同一脚本完成上述构造。
RSL-RL 路径通过 `src/unilab/rl/` 适配这个 tensor boundary。

多卡时每个 rank 按配置创建完整的 `algo.num_envs`、policy copy 与 rollout storage，
数据和 GAE 不跨 rank 交换。RSL-RL 负责 startup model broadcast、adaptive-KL 标量
同步，以及每个 PPO mini-batch 的梯度平均。UniLab 的 launcher 只负责 device/rank、
共享 log dir、seed offset 与进程失败联动，不另建 PPO 同步协议。只有 rank 0 写日志与
checkpoint；normalizer buffer、curriculum 和 episode 统计保持 rank-local。

### 异步 APPO 与 off-policy 路径

APPO 与 off-policy runner 采用环境到 learner 的拆分：

```text
Environment tensor loop -> shared IPC carrier -> learner
        ^                                      |
        +------------- SharedWeightSync -------+
```

- APPO 使用 `APPORunner`、`RolloutRingBuffer` 与 `SharedWeightSync`。
- SAC 与 FlashSAC 只使用一条 off-policy execution path：`ReplayBuffer`
  提供有界 host ingress，完整 ring 驻留在一个 CUDA/MPS learner device，
  `SharedWeightSync` 负责发布 actor 权重。
- `uni_rl.ipc.async_runner` (unilab-rl repo) 中的 `AsyncRunner` 负责 collector 进程启动、
  停止信号以及共享资源清理。

## 边界规则

- Manager-Based runtime 返回 `TorchEnvState`；CPU-authoritative 物理后端只在
  声明的 host-bridge H2D/D2H 边界交换，CUDA-native 后端保持 device-resident。
- optimizer state、rollout storage 等 trainer-only tensor 属于 learner 代码，
  而非 env 代码。
- Collector/learner 协议必须复用现有的 IPC 原语，而不是在 scripts 中另起临时的
  并行协议。
- PPO 多卡必须复用 RSL-RL 的 distributed contract；`algo.num_envs` 是 per-rank
  语义，不能在脚本中静默除以 world size。

## 仓库中的证据

- PPO 入口：`src/unilab/scripts/train_rsl_rl.py`
- APPO runner：`uni_rl.algos.appo.runner` (unilab-rl repo)
- Off-policy runner：`uni_rl.offpolicy.double_buffer_runner` (unilab-rl repo)
- IPC 原语：`uni_rl.ipc.async_runner` (unilab-rl repo)、
  `uni_rl.ipc.rollout_ring_buffer` (unilab-rl repo)、`uni_rl.ipc.replay_buffer` (unilab-rl repo)、
  `uni_rl.ipc.weight_sync` (unilab-rl repo)
