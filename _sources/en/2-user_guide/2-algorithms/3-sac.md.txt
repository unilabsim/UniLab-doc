# SAC

SAC runs through `src/unilab/scripts/train_sac.py`; FlashSAC has its own
entrypoint and per-algorithm config tree. The main config is
`src/unilab/conf/sac/config.yaml`, with the SAC algorithm defaults inlined there. The
current log name is `sac`.

## Runtime Model

The off-policy runner decouples simulation collection from accelerator learning through
bounded shared memory. A collector subprocess publishes packed transitions
through two ingress slots, while the complete replay ring is authoritative on
one CUDA or Apple MPS learner device. Host replay allocation therefore does not
grow with replay capacity; `ptr` and `size` advance only after a slot's device
copy completes. CUDA commits on a side stream, while MPS device work is
submitted only by the learner thread to avoid background Metal submission.
CPU and XPU training are unsupported; there is no alternate replay pipeline.

## Quick Start

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

## Key Fields

For the off-policy playback path (`src/unilab/scripts/train_sac.py` / CLI `--algo sac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=sac`
- `algo.num_envs=4096`
- `algo.batch_size=8192` is the learner batch per update.
- `algo.max_iterations=500`
- `training.use_amp=true` in `src/unilab/conf/sac/config.yaml`

The off-policy device replay path uses synchronized, learner-owned inference:
collectors exchange observations and actions through shared memory and do not own an actor.

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=1000 \
  training.no_play=true
```

## Single-node multi-GPU device placement

`CUDA_VISIBLE_DEVICES` is the sole GPU topology source. Each parent entry maps to one
rank, and within that rank the learner, collector, inference ring, replay ingress, and
backend payload all use the same physical GPU as local `cuda:0`. Off-policy collectors
never consume another rank's device or a parent-global CUDA ordinal.

IsaacGym, IsaacSim, and Genesis receive the rank-local simulator ordinal through the
environment override. Genesis binds its process-wide session before initialization.

MuJoCo has a committed multi-GPU scaling benchmark. The mjwarp per-rank placement contract is
covered by `tests/base/backend/test_process_device.py` and the off-policy runner/worker unit
tests; the repository does not currently contain an mjwarp multi-GPU throughput or convergence
benchmark.
