# UniLab

```{image} https://img.shields.io/badge/python-3.10%2B-blue
:alt: Python 3.10+
```
```{image} https://img.shields.io/badge/license-Apache--2.0-green
:alt: License
```

**UniLab** — *Universal Lab for Robot Learning.* Train robot RL without a GPU
simulation backend.

UniLab uses **CPU simulation + shared-memory runtime + GPU learning** instead of
coupling simulation and learning inside one GPU-resident pipeline.

```text
┌───────────────────┐                            ┌─────────────────────────┐
│  CPU Physics Sim  │   Unified Shared Memory    │   GPU Policy Training   │
│   MuJoCo/Motrix   │ ─────────────────────────▶ │     PPO / SAC / TD3     │
│ Multithread Step  │    SharedReplayBuffer      │ CUDA / MPS / ROCm / XPU │
└───────────────────┘                            └─────────────────────────┘
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Quick Demo
:link: user_guide/getting_started/quickstart
:link-type: doc

Install dependencies and run your first PPO training job in under three
minutes on a laptop.
:::

:::{grid-item-card} 📚 User Guide
:link: user_guide/index
:link-type: doc

Backends, algorithms, tasks, domain randomization, terrain — everything you
need to train policies.
:::

:::{grid-item-card} 🔁 Transfer Tutorials
:link: transfer/index
:link-type: doc

Sim-to-real on G1 / Go2 / Allegro, MuJoCo ↔ Motrix migration, and porting
tasks from Isaac Lab / Legged Gym.
:::

:::{grid-item-card} 🧩 API Reference
:link: api_reference/index
:link-type: doc

Detailed reference for every public class and function in `unilab` —
generated directly from the source tree.
:::

::::

## Why UniLab

- **No-GPU-sim**. The physics step never touches your GPU; the GPU is reserved
  for the learner. Works the same on Mac (MPS), NVIDIA (CUDA), AMD (ROCm),
  and Intel (XPU).
- **Multi-backend**. MuJoCo and Motrix share the same env / runner contract;
  backend choice lives in YAML, not in Python.
- **Contract-first**. Strict env / backend / runner / DR contracts so that
  PR-level changes can't silently break downstream tasks.
- **Cold-path asset access**. Hot loops never parse XML or probe backend
  internals — a discipline borrowed from real-time robotics stacks.

## Quick install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra motrix
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

For ROCm / Intel Arc / cluster setups see
{doc}`user_guide/getting_started/installation`.

## Get involved

- **Bugs & features** — [GitHub Issues](https://github.com/unilabsim/UniLab/issues)
- **Discussion** — [GitHub Discussions](https://github.com/unilabsim/UniLab/discussions)
- **Contributing** — start at {doc}`developer_guide/contributing`

## Citing UniLab

```bibtex
@misc{unilab2026,
  author = {UniLab Sim Authors},
  title  = {UniLab: A Universal Lab for Robot Learning},
  year   = {2026},
  url    = {https://github.com/unilabsim/UniLab}
}
```

```{toctree}
:hidden:
:caption: User Guide

user_guide/index
```

```{toctree}
:hidden:
:caption: Transfer

transfer/index
```

```{toctree}
:hidden:
:caption: Developer Guide

developer_guide/index
```

```{toctree}
:hidden:
:caption: API Reference

api_reference/index
```

```{toctree}
:hidden:
:caption: Reference

glossary
changelog
```
