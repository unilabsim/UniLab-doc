# WarpSAC

WarpSAC 通过 `src/unilab/scripts/train_warpsac.py` 运行，实现来自 unilab-rl
1.4.0 的 `uni_rl.algos.warp_sac`。它保留 FlashSAC 的 actor、分布式 critic、learner
与异步 double-buffer runtime，将均匀 replay 采样替换为按年龄线性偏置的分桶采样。

```bash
uv run train --algo warpsac --task g1_walk_flat --sim mujoco
uv run train --algo warpsac --task g1_motion_tracking --sim mjwarp
```

G1 的 MuJoCo 与 mjwarp owner 保持与 FlashSAC 对应配置相同的任务声明、policy I/O、
reward 与训练预算。WarpSAC 专属字段如下：

- `algo.decay_step`：线性近期偏好覆盖的年龄范围。
- `algo.replay_min_weight`：保留旧样本的最低采样权重。
- `algo.replay_num_buckets`：限制设备端偏置索引构建开销。
- `algo.actor_normalize_parameters` 与
  `algo.critic_normalize_parameters`：控制 FlashSAC 参数归一化。

WarpSAC 与 SAC/FlashSAC 一样要求 CUDA 或 Apple MPS 上的 device-resident replay
路径，日志写入 `logs/warp_sac/<task>/`。
