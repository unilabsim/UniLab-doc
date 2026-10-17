# Algorithms

Algorithm pages describe what each checked-in entrypoint runs, where its config
lives, and which command shape selects it. For general flags, see
{doc}`../1-training/1-cli_reference`.

| Algorithm | Style | Entrypoint | Config Evidence |
| --- | --- | --- | --- |
| PPO | synchronous on-policy | `src/unilab/scripts/train_rsl_rl.py` | `src/unilab/conf/ppo/config.yaml` |
| APPO | async on-policy | `src/unilab/scripts/train_appo.py` | `src/unilab/conf/appo/config.yaml` |
| SAC | off-policy | `src/unilab/scripts/train_sac.py` | `src/unilab/conf/sac/config.yaml` |
| FlashSAC | off-policy | `src/unilab/scripts/train_flashsac.py` | `src/unilab/conf/flashsac/config.yaml` |

Selection and log names are intentional: `--algo flashsac` composes
`algo.algo=flashsac`, records `algo.algo_log_name=flash_sac`, and imports the
implementation from `unilab.rl.algos.flash_sac`. `--algo ppo` keeps
`algo.algo=ppo`, records `algo.algo_log_name=rsl_rl_ppo`, and delegates to
upstream RSL-RL. PPO exposes `algo.obs_groups` as a list for its single policy
group; APPO, SAC, and FlashSAC use a mapping so policy and critic groups can
differ.

```{toctree}
:hidden:

1-ppo
2-appo
3-sac
4-flash_sac
```
