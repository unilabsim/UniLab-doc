# Transfer Tutorials

This section is a hands-on playbook for moving a UniLab policy across
*environments*, *backends*, or *frameworks*. Each tutorial follows the same
shape:

1. **What you start with** — the trained artefact and config.
2. **What changes** — the minimal set of edits in code, YAML, and assets.
3. **How you validate** — concrete commands and checkpoints.

```{toctree}
:caption: Sim-to-Real
:maxdepth: 1

sim_to_real/overview
sim_to_real/g1_whole_body
sim_to_real/go2_locomotion
sim_to_real/allegro_inhand
sim_to_real/onnx_export_and_runtime
sim_to_real/domain_randomization_for_real
sim_to_real/safety_layers
sim_to_real/latency_and_observation_lag
sim_to_real/troubleshooting
```

```{toctree}
:caption: Sim-to-Sim (MuJoCo ↔ Motrix)
:maxdepth: 1

sim_to_sim/why_switch
sim_to_sim/owner_yaml_swap
sim_to_sim/contact_and_friction_alignment
sim_to_sim/reward_parity_checks
sim_to_sim/playback_and_snapshot_differences
sim_to_sim/known_capability_gaps
```

```{toctree}
:caption: Framework Migration
:maxdepth: 1

framework_migration/from_isaac_lab
framework_migration/from_legged_gym
framework_migration/from_rsl_rl
framework_migration/from_skrl
framework_migration/task_config_translation
framework_migration/reward_porting_cookbook
```
