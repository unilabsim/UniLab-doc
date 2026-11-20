---
orphan: true
---

# ADR-0018 Collector Tensor Device Sole Placement Knob

语言: 简体中文

- Status: Proposed
- Date: 2026-10-21
- Owners: Training / RL runtime maintainers
- Supersedes: [ADR-0012](ADR-0012-sole-tensor-manager-and-scoped-backends.md) 的 2026-10-17（issue #2382）改名增补
- Superseded by: None
- Roadmap: [Issue #2481](https://github.com/Motphys/UniLab/issues/2481)

## Context

#2479 的归因收口确认：`training.sim_carrier_device`（env 载体驻留，仅 HOST_BRIDGE 路由，
`src/unilab/base/process_device.py`）与 `training.inference_transport`（off-policy IPC ring
放置）的合法组合是严格一维的——cpu↔host、cuda↔cuda，其余组合在
`src/unilab/rl/runtime/placement.py` 的 `resolve_inference_placement` 中 fail-closed。
两个旋钮表达一个语义，徒增非法组合面；#2479 重测中 CPU 载体 + CUDA ring 的组合即被
placement 正确拒绝，印证了一维约束。此外 `transport=cuda` 会反向强加
`env_tensor_native=True`，使配置语义依赖推导顺序，进一步说明 transport 不具备独立配置面。

ADR-0012 的 2026-10-17 增补（#2382）曾把 `training.collector_tensor_device` fail-closed
改名为 `training.sim_carrier_device`，理由是"键名大于语义"：该键当时只管 sim 载体，
管不到全部 collector tensor。旋钮合并后该理由消失——一维旋钮决定的正是 collector
tensor 的驻留设备（载体与 ring 同为派生事实），设备词值域（cpu/cuda）也优于运输层
词汇（host/cuda）。本 ADR 显式 supersede 该改名决策。

## Decision

1. `training.collector_tensor_device`（`auto`/`cpu`/`cuda`，默认 `auto`）是 collector
   tensor 驻留的唯一配置旋钮。`auto` 跟随已解析的 learner 设备：CUDA learner → `cuda`，
   CPU / torch.mps learner → `cpu`；显式 `cpu`/`cuda` 优先于 `auto`。
2. off-policy inference transport 降级为纯派生事实：placement 从
   `collector_tensor_device` 直接推导（`cpu` → host ring、`cuda` → CUDA ring），
   `training.inference_transport` 不再存在。`InferenceTransport` 枚举与 manifest
   词汇保留为运行时诊断，不构成配置面。
3. **彻底迁移，无兼容层**：`training.sim_carrier_device` 与
   `training.inference_transport` 的全部读取、校验、路由、Hydra YAML、测试与文档一并
   删除；按仓库先例为两个旧键保留一行"键已移除"的 fail-closed 报错（防呆守卫，非
   兼容层）。`select_sim_carrier_device`/`resolve_inference_transport` 收口为
   `select_collector_tensor_device`/`resolve_collector_tensor_placement`。
4. env 侧 owner 字段一并改名：`ManagerBasedRlEnvCfg.sim_carrier_device` →
   `collector_tensor_device`；`env.sim_carrier_device` 与 `env.manager_torch_device`
   一样带迁移报错 fail closed。`run_config.json` 的 `device_resolution` 记录键
   `sim_carrier` 改为 `collector_tensor`。
5. 删除 `transport=cuda → env_tensor_native=True` 的反向强加路径及其调用方依赖。
   one-env probe 不再参与 placement 推导，只做 fail-closed 交叉校验：probe 观测的
   tensor-native 事实必须与已解析 placement 一致，否则 env 未遵守路由，启动失败。

## Stable Contracts

| Contract | Owner / planned evidence |
| --- | --- |
| 单一旋钮与派生 transport | `src/unilab/rl/runtime/placement.py`、`tests/rl/runtime/test_placement.py` |
| 配置层解析与旧键 fail-closed | `src/unilab/training/tensor_runtime.py`、`tests/training/test_tensor_runtime.py` |
| HOST_BRIDGE 载体路由与旧 env 键守卫 | `src/unilab/base/process_device.py`、`tests/base/backend/test_process_device.py` |
| env owner 字段 | `src/unilab/envs/manager_based_rl_env.py`、`tests/envs/test_manager_based_rl_env.py` |
| owner YAML 键 | `src/unilab/conf/{sac,flashsac,ppo,appo}/config.yaml`、`tests/scripts/test_train_scripts.py` |
| probe 交叉校验 | `src/unilab/scripts/train_offpolicy.py`、`tests/algos/test_offpolicy_double_buffer_runner.py` |

## Alternatives Considered

- 保留 `training.inference_transport` 作为高级 override。Rejected：合法组合一维，
  第二旋钮只引入非法组合面与推导顺序依赖，没有任何可表达的新拓扑。
- 保留 `sim_carrier_device` 键名。Rejected：合并后旋钮语义就是 collector tensor 的
  驻留设备，`sim_carrier` 只描述 HOST_BRIDGE 载体这一段；且运输层词汇（host/cuda）
  不如设备词值域（cpu/cuda）直白。#2382 的"名大于义"理由在合并后不再成立。
- 为旧键保留一个 deprecation 期的双读兼容层。Rejected：违反仓库"一个含义一个公开名、
  无别名/deprecation fallback"的命名规则；fail-closed 迁移报错已给出可操作的迁移提示。

## Consequences

- 配置面降为一维：合法拓扑集合等于旋钮值域，非法组合在配置读取时即 fail closed。
- 使用 `training.sim_carrier_device` 或 `training.inference_transport` 的既有 owner
  YAML / CLI override 在启动时收到迁移报错；迁移动作是纯改名（值域不变）。
- `host`/`cuda` 的 transport 拼写不再是合法配置值；需要 CPU ring 的场景固定
  `training.collector_tensor_device=cpu`。
- run_config.json 的 `device_resolution.collector_tensor` 取代 `sim_carrier`；审计旧
  run 时需按 run 的创建版本解读记录键。
- 之后不得再引入第二个影响 collector tensor 驻留或 ring 放置的配置键；新的放置语义
  必须扩展该旋钮的值域或另立 ADR。

## Evidence In Repo

- 一维 placement 推导：`src/unilab/rl/runtime/placement.py::resolve_inference_placement`
- 配置层解析与旧键守卫：`src/unilab/training/tensor_runtime.py`
  （`select_collector_tensor_device`、`resolve_collector_tensor_placement`、
  `resolve_recorded_collector_tensor_device`）
- HOST_BRIDGE 路由与 env 键守卫：`src/unilab/base/process_device.py`
  （`apply_collector_tensor_device_override`、`canonicalize_collector_tensor_device`）
- env owner 字段：`src/unilab/envs/manager_based_rl_env.py::ManagerBasedRlEnvCfg.collector_tensor_device`
- owner YAML：`src/unilab/conf/sac/config.yaml`、`src/unilab/conf/flashsac/config.yaml`、
  `src/unilab/conf/ppo/config.yaml`、`src/unilab/conf/appo/config.yaml`
- 一维约束的测量来源：#2479 归因收口与受控重测；命名决策记录：#2481
- 被 supersede 的改名决策：ADR-0012 的 2026-10-17（#2382）增补

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0012 Sole Tensor Manager And Scoped Backends </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`
- {doc}`Tensor Runtime </zh_CN/2-user_guide/1-training/4-tensor_runtime>`
