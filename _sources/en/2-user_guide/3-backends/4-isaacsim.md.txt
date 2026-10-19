# IsaacSim Backend

> **IsaacSim is an explicit external-worker owner.** It requires the isolated
> Python 3.11 IsaacSim/IsaacLab runtime, a mapped entity scene, and same-GPU
> CUDA IPC. The first restored SAC owner is headless-training scoped.


UniLab's `isaacsim` backend runs IsaacSim 5.1.0 and IsaacLab v2.3.0 in a
dedicated Python 3.11 worker process. Pipe messages carry lifecycle metadata
while canonical qpos/qvel/control/reset arenas cross same-GPU CUDA IPC. The
current restored boundary is headless SAC `g1_walk_flat` training with a mapped
entity scene. Native interactive/record playback remains a separate runtime
follow-up and is not part of the training support claim.

## Runtime boundary

IsaacSim 5.1.0 is installed in a separate Python 3.11 environment because the
main UniLab environment supports Python 3.10--3.13. The setup entry point is
`scripts/tools/setup_isaacsim_env.sh`; it installs under
`$UNISIM_ISAACSIM_HOME` (default `$HOME/.cache/unisim/isaacsim`) and accepts the Kit
EULA through `OMNI_KIT_ACCEPT_EULA=1` for non-interactive worker startup.

The backend resolves these optional variables without importing Kit in the host
process:

- `UNISIM_ISAACSIM_HOME` selects the runtime root.
- `UNISIM_ISAACSIM_PYTHON` overrides the worker interpreter path.
- The former `UNILAB_ISAACSIM_HOME` and `UNILAB_ISAACSIM_PYTHON` names remain
  accepted as migration fallbacks.
- `OMNI_KIT_ACCEPT_EULA=1` keeps worker startup non-interactive.

The expected runtime layout is
`$UNISIM_ISAACSIM_HOME/venv/bin/python`, the Python 3.11 site-packages and
library directories under that venv, and an IsaacLab v2.3.0 source checkout at
`$UNISIM_ISAACSIM_HOME/IsaacLab`.

The render intent is part of the worker's cold `INIT` handshake. Training does
not inject a render mode and starts the inexpensive headless, camera-disabled
Kit experience. Eval selects one of these modes before Kit starts:

- `auto`: use the interactive Kit viewer when `DISPLAY` or `WAYLAND_DISPLAY`
  is present; otherwise use headless recording.
- `interactive`: start non-headless Kit and fail before worker launch when no
  display variable is present.
- `record`: start the headless rendering experience with IsaacLab RGB cameras;
  `training.play_steps` must be finite.
- `none`: run policy evaluation without a viewer or camera.

The record contract is RGB `(height, width, 3)`, `uint8`, contiguous, and
non-uniform. Invalid or placeholder frames fail closed instead of producing a
video. Width and height default to 1280 x 720 in the IsaacSim owner YAML and
can be overridden through `env.isaacsim_render_width` and
`env.isaacsim_render_height` before env creation.

Mapped scenes are cloned on a grid with a 2 m backend default. A finite,
positive `env.isaacsim_env_spacing` can widen that grid before env creation;
large task-local support surfaces should use more than twice the surface's
half-width so adjacent clones cannot overlap. IsaacSim reports the effective
spacing and environment origins back to the host, which rejects a mismatch.

The current worker supports MJCF materialization, batched articulation state,
position-target stepping, masked root/joint resets, a native Kit viewer, and
headless IsaacLab RGB camera capture. Contact-force sensors, reset or interval
domain randomization, and host pre-step callbacks remain unsupported and fail
closed.

Use the top-level CLI to select the backend and owner:

```bash
uv run train --algo sac --task g1_walk_flat --sim isaacsim
uv run eval --algo sac --task g1_walk_flat --sim isaacsim \
  --load-run <run-id> --render-mode record \
  training.play_steps=120 training.play_env_num=1 training.export_onnx=false
uv run eval --algo sac --task g1_walk_flat --sim isaacsim \
  --load-run <run-id> --render-mode interactive training.play_env_num=1
```

Record mode writes `play_video.mp4` in the selected run directory. These
commands require the external runtime and an NVIDIA CUDA device. The repository
has a headless SAC training completion record below, while native playback
remains a separate validation and is not claimed as stable.

## Training Validation

Issue #2387's semantic-quality follow-up completed one full headless SAC run on
the restored owner (RTX 4090, IsaacSim 5.1.0 / IsaacLab 2.3.0) after fixing
derived-sensor refresh and the mapped finite floor. It completed 5000/5000
iterations and 10,260,480 environment steps in 363.04 seconds of tracked
training wall time. Final/best mean reward was 291.201/298.049, mean episode
length was 993.99/1000, and final/tail throughput was 28,673/29,314 steps/s.

The reward and episode length reach the range of the validated MuJoCo-family
SAC runs for this canonical training workload. Native playback remains a
separate renderer validation and is not claimed here.

## Current Runtime Validation

A bounded SAC record eval and a bounded interactive eval using an existing
checkpoint were attempted on IsaacSim 5.1.0, IsaacLab v2.3.0, Kit 107.3.3,
Ubuntu 24.04.4, an RTX 4090, and NVIDIA driver 595.84. Both paths crashed during
`AppLauncher` initialization, before camera or viewer creation, with frames in
`librtx.scenedb.plugin.so`,
`libcarb.scenerenderer-rtx.plugin.so`, and `libomni.hydra.rtx.plugin.so` after
EGL initialization warnings. A minimal camera-enabled `AppLauncher` probe also
failed with `multi_gpu=False`.

This is a runtime blocker, not successful playback evidence. The backend keeps
the render protocol and its fail-closed tests; the SAC training cell is
`Tested`, but renderer/playback capability remains unpromoted. No placeholder
video is generated when the real renderer does not initialize.

## Inspecting The Contract

```bash
VIRTUAL_ENV="$HOME/.cache/unisim/isaacsim/venv" \
OMNI_KIT_ACCEPT_EULA=1 \
uv run --active --no-project \
  scripts/tools/probe_isaacsim_contract.py \
  --model-file src/unilab/assets/robots/g1/scene_flat.xml \
  --num-envs 2 --steps 2 --device cuda:0 \
  --output /tmp/isaacsim-contract.json
```

The command is a bounded developer probe. It only touches the XML/importer
during cold-path materialization and is useful for checking a newly installed
runtime; it is not a training or playback validation.

## Contract matrix

| UniLab contract | IsaacSim/IsaacLab operation | Observed result | Production constraint |
|---|---|---|---|
| MJCF scene materialization | `isaaclab.sim.converters.MjcfConverter` | G1 MJCF converts to USD successfully | Enable `isaacsim.asset.importer.mjcf` explicitly in headless workers |
| Batched articulation | `Articulation` + `ArticulationCfg` | 2 environments, 29 joints, 30 bodies | Resolve names at materialization; importer order is not the MJCF order |
| Quaternion layout | `robot.data.root_quat_w` / `body_quat_w` | `wxyz` | Keep `wxyz` at the shared-memory boundary |
| Base angular velocity | `robot.data.root_ang_vel_w` | World frame | Public getter remains world-frame; reset qvel conversion is a cold-path contract operation |
| Partial reset | `write_root_pose_to_sim`, `write_root_velocity_to_sim`, `write_joint_state_to_sim`, `reset(env_ids)` | Selected row changes; other row deltas are zero | Use masked batched writes; reject duplicate/out-of-range ids |
| Position control | `set_joint_position_target`, `write_data_to_sim`, `SimulationContext.step` | Target moves the first joint over bounded steps | `step(ctrl)` carries position targets; gains/limits are materialized explicitly |
| State getter boundary | `Articulation.data.*` tensors | All getters are batched with expected leading dimension | Worker copies tensors to host-owned shared-memory slots; hot getters do not parse assets |
| Rendering startup | `AppLauncher` cold mode selection | Mock worker verifies none/record/interactive mode, dimensions, and graphics handshake | Mode cannot change after env materialization |
| Offline RGB | IsaacLab `Camera` + `CameraCfg` | Protocol tests verify video writing and reject bad shape, dtype, or uniform frames; current real host crashes before camera creation | Require finite steps and keep renderer/playback claims unpromoted until real playback succeeds |
| Interactive viewer | non-headless Kit + `SimulationContext.set_camera_view` | Protocol tests drive a frame and map window close to `RenderClosedError`; current host has no successful bounded GUI evidence | Explicit interactive requires a display; `auto` falls back to record without one |
| Domain randomization | IsaacLab manager/event APIs | Not exercised | Non-empty unsupported plans must fail closed |

The importer returns a different joint/body ordering (for example, left/right
branches are interleaved). The worker builds name-to-index maps and reorders
every state/control array; positional assumptions would violate the
`SimBackend` index contract. The full owner and capability status is maintained
in {doc}`../../5-reference/5-support_matrix`.
