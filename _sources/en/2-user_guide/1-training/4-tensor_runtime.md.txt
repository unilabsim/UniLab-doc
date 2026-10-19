# Tensor runtime

SAC and FlashSAC resolve their spawn-facing tensor-runtime settings
before probing the environment or constructing a learner. The owner YAML values
are collected into one bounded `TensorRuntimeSettings` object and passed to the
off-policy runner. Invalid values, unsupported bounds, and impossible memory
combinations therefore fail before collector or learner processes are launched.

This contract covers the scoped single-GPU off-policy runtime. It is not an
autotuner and does not add multi-GPU scaling.

## Owner defaults and bounds

| Setting | Default | Valid range or constraint |
| --- | ---: | --- |
| `training.inference_slot_capacity` | `1` | Positive integer, at most `16` |
| `training.collector_metrics_interval` | `1` | Positive integer, at most `10000` |
| `training.replay_ingress_depth` | `2` | Positive integer, at most `16` |
| `training.replay_ingress_slot_rows` | `null` | `1` through `algo.num_envs`; `null` means `algo.num_envs` |
| `training.cuda_process_sharing` | `null` | `null` or explicit `mps` for single-rank MJWarp |
| Learner rows per synchronization | `algo.batch_size * algo.updates_per_step` | Constrained by the CUDA memory budget |

Values other than `training.cuda_process_sharing` must be exact positive
integers. Booleans, strings, floating-point numbers, zero, and negative values
fail closed. `training.cuda_process_sharing` accepts only `null` or the explicit
string `mps`. The G1 Motion Tracking / MJWarp owner intentionally overrides only
`collector_metrics_interval` to `100`; all other tensor-runtime defaults remain
as listed above.

## Device semantics and resolution

Device-related settings are separated by the layer that consumes them:

| Configuration key | Effective layer | Valid values |
| --- | --- | --- |
| `training.sim_carrier_device` | Training assembly routes it to the environment | `auto`, `cpu`, or unindexed `cuda` |
| `env.sim_carrier_device` | Manager public tensors for a MuJoCo HOST_BRIDGE | `cpu`, `cuda`, or `cuda:<index>` |
| `training.inference_transport` | Off-policy IPC placement | `host`, `cuda`, or `null` (auto) |
| `training.device` | Learner and rank-local process selection | Concrete Torch device label |

The training key is the only user-facing carrier request. `auto` follows the
resolved learner device: CUDA routes the rank-local indexed device, while CPU
and torch.mps (Metal Performance Shaders) keep the CPU carrier. Explicit
`cpu` or `cuda` wins over `auto`. `cuda` is concrete and never maps to MPS; a
CUDA request on a host without CUDA fails closed.

`cuda` is supported only for the MuJoCo HOST_BRIDGE carrier. Other backends
ignore `auto`/`cpu`, but a `cuda` request fails closed. `run_config.json`
retains the composed configuration and records `run.resolved_devices` with
each configured-to-resolved mapping; startup logs print the same mapping.

Removed keys fail closed with a migration error: use `training.sim_carrier_device`
instead of `training.collector_tensor_device`, `env.sim_carrier_device` instead
of `manager_torch_device`, and the resolved learner device instead of
`env.tensor_runtime_device`.

Migration note: a CUDA learner with the default `auto` presents CUDA tensors
through the packed host bridge and resolves CUDA inference transport. Each
collector process then holds one additional CUDA context on the learner GPU
(see {doc}`7-tensor_runtime_production`). Pin
`training.sim_carrier_device=cpu` to keep the CPU carrier.

For off-policy inference, `host` names the transport boundary rather than a
complete learner topology: observations and actions use the CPU inference ring,
while the learner itself may be CPU or Apple MPS. This spelling serves pure CPU
tests and the macOS MPS host-boundary cell. `cuda` follows Torch device-type
semantics and covers both NVIDIA CUDA and AMD ROCm. It does not relax separate
platform gates: CUDA-only tensor backends still fail closed on ROCm/HIP, and
`training.cuda_process_sharing` remains available only to the NVIDIA-specific
CUDA MPS daemon and native H2D implementation. The removed `cpu` spelling fails
closed; use `host`.

The word “worker” is likewise layer-specific:

| Term | Meaning |
| --- | --- |
| Collector worker | APPO or off-policy process that steps environments and publishes rollouts |
| Backend worker | Backend-owned external simulator process (for example Isaac) |
| torchrun rank | Distributed learner rank launched by the PPO multi-GPU launcher |
| Backend process device | Process-global accelerator binding, not a Manager carrier key |

## Replay ingress tradeoffs

The default `replay_ingress_slot_rows: null` resolves to `algo.num_envs` and
publishes one ingress chunk per collector vector. This keeps the publication
and synchronization pattern predictable and is the recommended production
default.

Setting the field to a smaller value reduces resident ingress memory by splitting
one collector vector into contiguous chunks. It is intended only for explicit
memory-pressure tuning. More chunks mean more publications, more opportunities
for producer backpressure, and—on CUDA—one current-stream barrier per published
chunk. Always benchmark a smaller value on the target workload before adopting
it.

Ingress diagnostics count publication chunks, not complete logical collector
vectors. A shutdown may retain a valid row prefix of a partially published
vector; it never publishes an invalid transition row. See {doc}`3-logging` for
the replay-ingress counters and shutdown schema.

## Random number generation

The Manager environment owns a `unilab.managers.TorchManagerRng` as its sole
training-runtime RNG. This adapter wraps one explicit CPU or CUDA
`torch.Generator`, validates seeds and unsupported sampling semantics, and
returns tensors on that generator's device. It is not a NumPy
bitstream-compatibility layer. The training-entry seed contract seeds global
Python and Torch RNGs only; it no longer mutates the legacy global NumPy RNG.

The scoped G1 Motion Tracking / FlashSAC owner uses a CUDA `TorchManagerRng` for
observation corruption, adaptive/mixed motion sampling, and selected-row reset.
A/B evaluation therefore compares reproducibility and distributions under the
same integer seed, not bitwise NumPy trajectories. Use an explicit `env.seed`
for deterministic runs; `algo.seed` seeds the training run's global Python and
Torch RNGs but does not replace that owner-local generator seed.

`TorchManagerRng.get_state()` returns the generator's CPU state tensor and
`set_state()` restores it. The exact generator state remains the authoritative
reproducibility payload; an owner that restores or audits RNG records that
state tensor. The runtime keeps no per-transition version accounting.

## Runtime evidence

The runtime manifest records the effective evidence needed to audit a run:

- `runtime_limits` contains configured, default, effective, and maximum values
  for the resolved tensor-runtime knobs.
- `inference_memory_budget` records the bounded CUDA inference-ring budget.
- `tensor_memory_budget` records the combined CUDA inference, replay storage,
  replay ingress, learner batch, and workspace budget calculation.
- `cuda_process_sharing` records validated execution-sharing evidence when the
  owner explicitly requests `mps`; it is absent for the default `null` mode.

The manifest is written before spawn when a budget decision is made. If an
unsafe combination is rejected, the error identifies the offending setting and
the process is not launched.

For the single-GPU G1 FlashSAC/MJWarp production workflow, benchmark
reproduction, artifacts, and troubleshooting, see
{doc}`7-tensor_runtime_production`.
