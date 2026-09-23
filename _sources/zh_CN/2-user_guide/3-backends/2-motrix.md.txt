# Motrix 后端

Motrix 是一个可选后端，通过 `motrix` extra 安装。该 extra 委托给
`unisim-core[motrix]`，runtime 版本固定在 UniSim 的 `pyproject.toml`
中，适配层位于 `unisim.backend.motrix` 下。

## 安装

```bash
uv sync --extra motrix
```

`make setup-motrix` 会执行相同的依赖同步，并安装 shell 自动补全。

## 何时使用

- task owner 以 `src/unilab/conf/.../<task>/motrix.yaml` 形式存在。
- 你想要 Motrix 原生交互式回放；该后端提供原生交互式渲染器和视频录制能力。
- 所生成的支持矩阵将你的 entrypoint/task/backend 组合标记为 configured 或 tested。

## 命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

使用 `--render-mode record` 进行无头的仅视频回放。后端选择请保留在
`--sim motrix` 中，而不要单独 override `training.sim_backend`。

## CPU 亲和性

`training.dp_collector_cpu_ids` 可以为每个 collector rank 提供一个显式
CPU-id 列表。该列表以 `EnvCfg.cpu_ids` 传入环境，并在冷路径上校验：
条目必须非空、去重、非负且对所属进程可用。随后 Motrix 适配层在首次
模型加载之前将 MotrixSim 的共享 worker 线程池绑定到这些核上 —— worker
`i` 使用 `cpu_ids[i % len(cpu_ids)]` —— 环境构建同时把所属进程限制在
同一 CPU 块内，使 step 后的宿主机计算留在该 rank 的分区内。默认值
`null` 保持 MotrixSim 默认线程池策略和操作系统调度不变。校验后的 CPU
块可通过后端只读属性 `cpu_ids` 查询。
