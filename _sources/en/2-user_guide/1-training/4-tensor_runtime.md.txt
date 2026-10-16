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

## Collector tensor device (MuJoCo host bridge)

`training.collector_tensor_device` selects the optional accelerator Torch
carriers of the MuJoCo HOST_BRIDGE Manager (routed to the
`manager_torch_device` owner field). It accepts `auto` (the default), `cpu`,
or unindexed `cuda`; indexed values such as `cuda:3` are rejected.

- `auto` follows the learner device: a CUDA learner routes the rank-local
  indexed learner device (for example `cuda:3`), so collector observations and
  actions stay on the learner GPU and the runtime probe
  (`unilab.rl.contracts.probe`) resolves CUDA inference
  transport with no further configuration. CPU and torch.mps (Metal Performance
  Shaders) learners keep the CPU carrier.
- Explicit `cpu` or `cuda` values win over `auto`. `cuda` remains supported
  only for the MuJoCo HOST_BRIDGE backend; for every other backend the knob is
  a no-op unless it requests `cuda`, which fails closed.

Migration note: existing MuJoCo configurations that ran a CUDA learner under
the previous `cpu` default now present CUDA tensors through the packed host
bridge and resolve CUDA inference transport. Each collector process then holds
one additional CUDA context on the learner GPU (see
{doc}`7-tensor_runtime_production` for the two-context topology). Pin
`training.collector_tensor_device=cpu` to keep the previous CPU carrier.

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
