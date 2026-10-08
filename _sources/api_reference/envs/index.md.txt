# `unilab.envs` — Environment runtime

Task-agnostic Manager-Based environment runtime and reusable MDP terms.
Concrete task implementations are owned by {doc}`../tasks/index`.

`ManagerBasedRLEnv` exposes the Torch public lifecycle while executing
community-style action, observation, reward, termination, event, command, and
curriculum managers through the explicit P1 NumPy host boundary.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.envs
```
