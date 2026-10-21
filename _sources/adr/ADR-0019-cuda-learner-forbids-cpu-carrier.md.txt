---
orphan: true
---

# ADR-0019 CUDA Learner Forbids CPU Carrier

语言: 简体中文

- Status: Proposed
- Date: 2026-10-21
- Owners: Training / RL runtime maintainers
- Supersedes: None
- Superseded by: None
- Roadmap: [Issue #2483](https://github.com/Motphys/UniLab/issues/2483)

## Context

#2479 的归因重测确认：CUDA learner 下 CPU 载体与 CUDA 载体 + CUDA MPS 吞吐持平
（87.4k vs 88.4k steps/s，噪声内）。CPU 载体的存在价值已从"性能路径"退化为"平台
兜底"，但它仍作为合法配置出现在 CUDA learner 路径上，使新跑法可以静默选入一个
无性能收益、徒增 host 边界拷贝的拓扑。

ADR-0018 把 collector tensor 驻留收口为单一旋钮
`training.collector_tensor_device`；本 ADR 在该一维旋钮上收紧值域的合法组合：
`cpu` × CUDA learner 不再是默认合法组合。

## Decision

1. **CUDA learner 禁止显式 CPU 载体，fail closed**。两个执行面同步收紧：
   - off-policy placement：`src/unilab/rl/runtime/placement.py` 的
     `resolve_inference_placement` 在解析出 `cpu` 请求且 learner 为 CUDA 设备族时
     抛出带迁移提示的 `ValueError`；
   - env 载体路由：`src/unilab/base/process_device.py` 的
     `apply_collector_tensor_device_override` 在显式 `cpu`、已知 CUDA learner、
     HOST_BRIDGE 后端时同样 fail closed（on-policy 与 play 路径经此覆盖）。
   `auto` 不受影响：CUDA learner 的 `auto` 本来就解析为 `cuda`。
2. **显式逃生门是环境变量，不是配置键**。归因基线/调试 run 设置
   `UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER=1`（真值集 `1/true/yes/on`）显式 opt-in；
   读取点收口在 `src/unilab/process_context.py` 的
   `cuda_learner_cpu_carrier_opted_in`，训练配置层把它绑定到 placement 的
   `allow_cpu_carrier_with_cuda_learner` 参数。不新增 Hydra 键，保持 ADR-0018 的
   单一旋钮配置面；逃生门是纯部署层设施，不进 owner YAML。
3. **逐条豁免依据**：
   - **CPU learner**：learner 设备族为 `cpu`，host ring 是唯一拓扑，无设备边界
     可跨越；规则以 CUDA 设备族为触发条件，天然不命中。
   - **macOS torch.mps learner**：learner 设备族为 `mps`，同样不命中设备族检查。
     Apple Silicon 上没有 CUDA IPC/MPS 等价物，CPU 载体是 mps learner 唯一可用的
     host bridge 拓扑（staging policy
     `cpu_ring_explicit_learner_device_copy`）。
   - **ROCm/HIP**：ROCm PyTorch 使用 `cuda` 设备命名空间，单看设备族会误伤；以
     `torch.version.hip` 判定 HIP 构建后豁免。ROCm 没有 CUDA MPS 等价物
     （`training.cuda_process_sharing` 仅限 NVIDIA），CPU 载体是其唯一的进程共享
     缓解路径。

## Stable Contracts

| Contract | Owner / planned evidence |
| --- | --- |
| CUDA learner + 显式 `cpu` fail closed | `src/unilab/rl/runtime/placement.py`、`tests/rl/runtime/test_placement.py` |
| 配置层绑定逃生门 | `src/unilab/training/tensor_runtime.py`、`tests/training/test_tensor_runtime.py` |
| env 载体路由同步收紧 | `src/unilab/base/process_device.py`、`tests/base/backend/test_process_device.py` |
| 逃生门环境变量读取 | `src/unilab/process_context.py`（`CUDA_LEARNER_CPU_CARRIER_OPT_IN`） |
| 入口端到端行为 | `tests/scripts/test_train_scripts.py` |
| 支持矩阵配对契约文字 | `scripts/tools/support_matrix.py`、`tests/scripts/test_support_matrix.py` |

## Alternatives Considered

- 直接删除 CPU 载体（无逃生门）。Rejected：#2479 的归因基线需要可重复地跑出
  CPU 载体 + CUDA learner 拓扑作为对照臂；ROCm/macOS/CPU learner 也仍依赖该路径。
- 逃生门做成 Hydra 配置键（如 `training.allow_cuda_learner_cpu_carrier`）。
  Rejected：会在 ADR-0018 刚收口的单一旋钮旁重新长出一个永久性配置面；环境变量
  把逃生门限定在部署/调试层，且不会让 owner YAML 表达出一个被禁止的拓扑。
- 只在 off-policy placement 一层收紧。Rejected：on-policy（PPO/APPO）与 play 路径
  经 `apply_collector_tensor_device_override` 同样可以选出 CPU 载体 + CUDA
  learner；单层收紧会让契约依入口而不一致。

## Consequences

- `training.collector_tensor_device=cpu` 与非 ROCm CUDA learner 的组合在配置读取/
  路由时即报错，报错文本给出 `auto`/`cuda` 迁移路径与逃生门变量名。
- 归因基线 run 通过 `UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER=1` 复现旧拓扑；其
  run_config.json 中的 `device_resolution.collector_tensor=cpu` 与
  `staging_policy=cpu_ring_explicit_learner_actor_h2d_action_d2h` 仍是识别该拓扑的
  审计证据。
- 支持矩阵的 HOST_BRIDGE `cpu`/`cuda` Torch devices 列描述的是后端能力清单，不因
  本契约改变；矩阵正文新增 pairing contract 说明，把"能力存在"与"CUDA learner
  下默认禁用"区分开。
- 之后不得引入绕过该禁令的第二逃生通道；新的豁免平台必须扩展本 ADR 的豁免清单
  并给出逐条依据。

## Evidence In Repo

- 禁令与逃生门参数：`src/unilab/rl/runtime/placement.py::resolve_inference_placement`
- 配置层绑定：`src/unilab/training/tensor_runtime.py::resolve_collector_tensor_placement`
- env 载体路由：`src/unilab/base/process_device.py::apply_collector_tensor_device_override`
- 逃生门读取：`src/unilab/process_context.py::cuda_learner_cpu_carrier_opted_in`
- 吞吐持平的测量来源：#2479 归因收口与三臂基线；契约决策记录：#2483

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0018 Collector Tensor Device Sole Placement Knob </adr/ADR-0018-collector-tensor-device-sole-placement-knob>`
- {doc}`Tensor Runtime </zh_CN/2-user_guide/1-training/4-tensor_runtime>`
