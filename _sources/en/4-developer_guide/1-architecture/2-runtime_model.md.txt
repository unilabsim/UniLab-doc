# Runtime Model

The detailed runtime contract is in
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries` and
{doc}`/zh_CN/4-developer_guide/0-index`. This page keeps the English
summary close to the code paths.

## Runtime Shapes

### Synchronous PPO Paths

`src/unilab/scripts/train_rsl_rl.py` composes Hydra config,
calls registry bootstrap, constructs the strict `TorchEnv` through
`registry.make(...)`, and runs the learner in the same process. The RSL-RL path
adapts that tensor boundary through `src/unilab/rl/`.

### Async APPO And Off-Policy Paths

APPO and off-policy runners use an environment-to-learner split:

```text
Environment tensor loop -> shared IPC carrier -> learner
        ^                                      |
        +------------- SharedWeightSync -------+
```

- APPO uses `APPORunner`, `RolloutRingBuffer`, and `SharedWeightSync`.
- SAC and FlashSAC use one off-policy execution path: `ReplayBuffer`
  provides bounded host ingress, the complete ring lives on one CUDA/MPS
  learner device, and `SharedWeightSync` publishes actor weights.
- `AsyncRunner` in `uni_rl.ipc.async_runner` (unilab-rl repo) owns collector process
  startup, stop signaling, and shared-resource cleanup.

## Boundary Rules

- The Manager-Based runtime returns `TorchEnvState`; CPU-authoritative physics
  backends cross their declared host-bridge H2D/D2H boundaries, while
  CUDA-native backends remain device-resident.
- Trainer-only tensors such as optimizer state and rollout storage belong to
  learner code, not env code.
- Collector/learner protocols must reuse the existing IPC primitives instead of
  creating ad-hoc parallel protocols in scripts.

## Evidence In Repo

- PPO entrypoint: `src/unilab/scripts/train_rsl_rl.py`
- APPO runner: `uni_rl.algos.appo.runner` (unilab-rl repo)
- Off-policy runner: `uni_rl.offpolicy.double_buffer_runner` (unilab-rl repo)
- IPC primitives: `uni_rl.ipc.async_runner` (unilab-rl repo),
  `uni_rl.ipc.rollout_ring_buffer` (unilab-rl repo), `uni_rl.ipc.replay_buffer` (unilab-rl repo),
  `uni_rl.ipc.weight_sync` (unilab-rl repo)
