# `unilab.training` — Training Runtime

Glue between `algos`, `envs` and `ipc`: experiment lifecycle and the
top-level `run` helpers invoked by the `train` / `eval` / `demo` CLI
entrypoints. Training seeding lives in `unilab.training.seed`; layer-0
helpers (reward bookkeeping, checkpoint resolution, sim2sim contracts)
live in `unilab.utils`, and resolved env config adaptation lives in
`unilab.base.config_adapter`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.training
```
