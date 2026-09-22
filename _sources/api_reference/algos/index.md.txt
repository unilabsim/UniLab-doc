# Learning Algorithms — `uni_rl` plus the in-repo PPO integration

Most of the RL algorithm layer moved out of the `unilab` package into the
independently released **uni_rl** package (distribution name `unilab-rl`,
published on PyPI; issue #1480):

- `uni_rl.algos.appo` — APPO runner, learner, staging, worker
- `uni_rl.algos.fast_sac` / `uni_rl.algos.fast_td3` / `uni_rl.algos.flash_sac` — off-policy learners and runners
- `uni_rl.offpolicy` — generic off-policy runner, worker, thread budget
- `uni_rl.algos.common` — shared actor factory, networks, normalization, compile helpers

The PPO (RSL-RL) integration lives in UniLab itself: `unilab.rl`
provides the `RslRlVecEnvAdapter` plus distributed/seed helpers, and
`src/unilab/scripts/train_rsl_rl.py` drives upstream
`rsl_rl.runners.OnPolicyRunner` / `rsl_rl.algorithms:PPO` directly.

UniLab keeps the training *entrypoints* (`src/unilab/scripts/train_*.py`),
which inject environments into uni_rl runners through
`uni_rl.env_contract.EnvFactory`; see `src/unilab/base/env_factory.py` for the
registry-backed adapter.

All trainers conform to a single runner contract — see
{doc}`../../en/4-developer_guide/2-contracts/5-runner_lifecycle`.
