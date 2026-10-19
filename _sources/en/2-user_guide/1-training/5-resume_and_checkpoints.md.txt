# Resume And Checkpoints

Checkpoint selection is controlled by algorithm-level fields. Use
`algo.load_run`, not `training.load_run`.

## Resume Training

PPO/APPO resume from `algo.load_run` directly. Use a run id or `-1` for the
latest run in the relevant log directory:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true
```

Off-policy algorithms (SAC/FlashSAC/WarpSAC) keep resume opt-in so that the
`algo.load_run=-1` default can never turn a fresh launch into an accidental
resume. Set `algo.resume=true` and select the checkpoint with `algo.load_run`
(run id or `-1` for the latest run) plus optional `algo.checkpoint`
(iteration or filename; `-1` selects the latest checkpoint):

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp \
  --profile <owner-profile> \
  algo.resume=true \
  algo.load_run=2026-10-05_14-28-04_mjwarp \
  training.no_play=true
```

The off-policy checkpoint restores the full learner state (networks,
optimizers, schedulers, normalizers, and update count), and training
continues with absolute iteration numbering (`model_32000.pt` resumes at
iteration 32001, so `algo.max_iterations` stays the total budget, not an
increment). The replay buffer is not checkpointed: a resumed run re-warms
from an empty buffer through the usual train-start threshold, and RNG state
is not restored.

## Checkpoint Format Support Boundary

PPO/APPO resume loads UniLab-patched rsl-rl checkpoints. A supported
checkpoint carries `unilab_logger_state` (the logger `tot_time` /
`tot_timesteps` counters, required) and, when the task environment publishes
versioned training state, a matching `unilab_env_training_state`. UniLab has
written both keys since the resume-state patch (issue #441, shipped since
v1.0.0), and new checkpoints keep writing them unchanged.

Starting with the first UniLab release after 1.3.5 that includes issue #2416,
loading fails closed:

- a checkpoint without a valid `unilab_logger_state` (a pre-patch or vanilla
  rsl-rl checkpoint) is rejected with an error naming the minimum supported
  format, instead of silently restarting training progress from 0;
- when the environment publishes training state, a checkpoint without a valid
  versioned `unilab_env_training_state` is rejected as well;
- an environment without exportable training state writes no placeholder key
  and requires none.

There is no compatibility window, offline migration tool, or dual-write path,
and the fail-closed behavior is not backported to a 1.3.5 patch release. To
recover an unsupported pre-patch checkpoint, retrain with a supported UniLab
version or resume from a checkpoint saved by one.

Checkpoint files are immutable load inputs: the loader never repairs,
rewrites, or migrates them.

## Replay A Checkpoint

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` maps `--load-run` to the underlying checkpoint selector and sets
playback mode:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

Some script paths accept a checkpoint path through `algo.load_run`; the unified
CLI validates `--load-run` as a run id and does not accept path separators.

## Seeds

Training seed resolution is implemented in `src/unilab/training/seed.py`.
Only the algorithm-level `algo.seed` is honored; the former `training.seed`
fallback was removed, and a config that sets `training.seed` without
`algo.seed` fails closed. The helper records seed metadata when experiment
tracking is active.
