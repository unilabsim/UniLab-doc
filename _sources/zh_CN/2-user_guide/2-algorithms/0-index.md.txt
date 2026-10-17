# 算法

算法页面描述每个内置入口运行的内容、其配置所在位置，以及用哪种命令形式来选择它。
关于通用 flag，请参见 {doc}`../1-training/1-cli_reference`。

| 算法 | 类型 | 入口 | 配置证据 |
| --- | --- | --- | --- |
| PPO | 同步 on-policy | `src/unilab/scripts/train_rsl_rl.py` | `src/unilab/conf/ppo/config.yaml` |
| APPO | 异步 on-policy | `src/unilab/scripts/train_appo.py` | `src/unilab/conf/appo/config.yaml` |
| SAC | off-policy | `src/unilab/scripts/train_sac.py` | `src/unilab/conf/sac/config.yaml` |
| FlashSAC | off-policy | `src/unilab/scripts/train_flashsac.py` | `src/unilab/conf/flashsac/config.yaml` |

选择名与日志名是有意区分的：`--algo flashsac` 组合
`algo.algo=flashsac`，记录 `algo.algo_log_name=flash_sac`，并从
`unilab.rl.algos.flash_sac` 导入实现。`--algo ppo` 保持
`algo.algo=ppo`，记录 `algo.algo_log_name=rsl_rl_ppo`，并直接委托上游
RSL-RL。PPO 的 `algo.obs_groups` 是单一 policy group 的列表；APPO、SAC 与
FlashSAC 使用 mapping，以便 policy/critic group 可以不同。

```{toctree}
:hidden:

1-ppo
2-appo
3-sac
4-flash_sac
```
