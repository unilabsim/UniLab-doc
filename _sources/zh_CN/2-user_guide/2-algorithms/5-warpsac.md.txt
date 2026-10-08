# WarpSAC

WarpSAC 通过 `src/unilab/scripts/train_warpsac.py` 运行，实现来自 unilab-rl
1.5.0 的 `uni_rl.algos.flash_sac`。它保留公开的 WarpSAC 配置身份，并选择
FlashSAC 的年龄偏置 replay 选项。

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

WarpSAC 与 SAC/FlashSAC 共享同一组公开 tensor-runtime 设置。默认值、边界、
replay ingress 权衡以及 manifest 证据见
{doc}`../1-training/4-tensor_runtime`。
