# Shared-Memory Runtime — `unilab.rl.runtime`

The async IPC layer lives in the `unilab` distribution under
`unilab.rl.runtime` (the former `uni_rl.ipc` / `uni_rl.offpolicy` modules are
merged into this repository; issue #2372): async runner, shared-memory
buffers, replay pipelines, inference exchange, DP launcher / sync, and weight
sync.

| Submodule | Role |
|---|---|
| `unilab.rl.runtime.async_runner` | The high-level orchestration loop |
| `unilab.rl.runtime.ipc.tensor_rollout_ring_buffer` | Tensor-backed rollout window used by APPO collectors |
| `unilab.rl.runtime.ipc.replay_buffer` | Bounded shared ingress for off-policy transitions |
| `unilab.rl.runtime.ipc.replay_pipelines.*` | Authoritative CUDA/MPS replay ring, device gather, and native H2D |
| `unilab.rl.runtime.ipc.inference_ring` | Bounded, ordered observation/action exchange for learner-owned off-policy inference |
| `unilab.rl.runtime.ipc.weight_sync` | Push learner weights to on-policy collector workers |
| `unilab.rl.runtime.distributed.*` | torchrun launch, rank context, CPU affinity, and DP parameter sync |
