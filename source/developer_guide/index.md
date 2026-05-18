# Developer Guide

If you are submitting a PR, extending UniLab with a new backend / task /
algorithm, or auditing the runtime architecture — start here.

```{toctree}
:caption: Onboarding
:maxdepth: 1

contributing
contributing_workflow
```

```{toctree}
:caption: Architecture
:maxdepth: 1

architecture/development_standard
architecture/runtime_model
architecture/layer_boundaries
architecture/scene_composition
architecture/registry_bootstrap
```

```{toctree}
:caption: Contracts
:maxdepth: 1

contracts/env_contract
contracts/backend_capability
contracts/task_owner_config
contracts/domain_randomization
contracts/runner_lifecycle
```

```{toctree}
:caption: Extending UniLab
:maxdepth: 1

extending/new_task
extending/new_backend
extending/new_algorithm
extending/new_terrain
```

```{toctree}
:caption: Architecture Decision Records
:maxdepth: 1

adr/README
adr/ADR-0001-runtime-model-and-layer-boundaries
adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot
adr/ADR-0003-task-owner-and-config-compose-contract
adr/ADR-0004-registry-bootstrap-contract
adr/ADR-0005-unified-obs-critic-env-and-ipc-contract
adr/ADR-TEMPLATE
```
