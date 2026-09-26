# WarpSAC

WarpSAC runs through `src/unilab/scripts/train_warpsac.py` and is implemented
by `uni_rl.algos.warp_sac` from unilab-rl 1.4.0. It keeps the FlashSAC actor,
distributional critic, learner, and asynchronous double-buffer runtime, while
replacing uniform replay sampling with a bucketed linear age-bias sampler.

```bash
uv run train --algo warpsac --task g1_walk_flat --sim mujoco
uv run train --algo warpsac --task g1_motion_tracking --sim mjwarp
```

The G1 MuJoCo and mjwarp owners keep the same task declarations, policy I/O,
rewards, and training budgets as their FlashSAC counterparts. WarpSAC-specific
fields select the replay regime:

- `algo.decay_step` is the age span for the linear recency bias.
- `algo.replay_min_weight` retains coverage of older transitions.
- `algo.replay_num_buckets` bounds biased-index construction on device.
- `algo.actor_normalize_parameters` and
  `algo.critic_normalize_parameters` control FlashSAC parameter normalization.

WarpSAC requires the same CUDA or Apple MPS device-resident replay path as
SAC and FlashSAC, and writes runs under `logs/warp_sac/<task>/`.
