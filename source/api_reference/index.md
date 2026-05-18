# API Reference

Detailed API reference for the `unilab` Python package. Every public class,
function, and submodule is auto-extracted from the source tree via
`sphinx.ext.autodoc` + `sphinx.ext.autosummary` — if you're reading this
online and a symbol you expect is missing, please open an issue.

## How to read this section

- **`base/`** — the contracts that everything else depends on. `NpEnv`,
  `SimBackend`, `Registry`, `Scene` live here.
- **`envs/`** — concrete tasks (locomotion / manipulation / motion tracking)
  layered on top of `base`.
- **`algos/`** — PPO, SAC, TD3 and their variants, in both PyTorch and MLX.
- **`backend/`** — MuJoCo and Motrix adapters.
- **`ipc/`** — shared-memory primitives that connect simulator workers to
  the GPU learner.
- **`training/`** — runtime helpers, monitoring, reward bookkeeping.
- **`dr/`, `terrains/`, `visualization/`** — orthogonal subsystems.

## Top-level modules

```{toctree}
:maxdepth: 1

base/index
envs/index
algos/index
backend/index
ipc/index
training/index
dr/index
terrains/index
visualization/index
tools/index
utils/index
logging/index
top_level
```
