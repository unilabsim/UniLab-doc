# Tensor runtime

SAC, FlashSAC, and WarpSAC resolve their spawn-facing tensor-runtime settings
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

The generic Manager seam remains NumPy-owned: `ManagerBasedRlEnv.rng` is an
`np.random.Generator`, and existing seeded streams stay unchanged. Narrow
device-resident owners may instead own a `unilab.managers.TorchManagerRng`.
This adapter wraps one explicit CPU or CUDA `torch.Generator`, validates seeds
and unsupported sampling semantics, and returns tensors on that generator's
device. It is not a NumPy bitstream-compatibility layer.

The scoped G1 Motion Tracking / FlashSAC owner uses a CUDA `TorchManagerRng` for
observation corruption, adaptive/mixed motion sampling, and selected-row reset.
A/B evaluation therefore compares reproducibility and distributions under the
same integer seed, not bitwise NumPy trajectories. Use an explicit `env.seed`
for deterministic runs; `algo.seed` seeds the training run's global Python,
NumPy, and Torch RNGs but does not replace that owner-local generator seed.

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
