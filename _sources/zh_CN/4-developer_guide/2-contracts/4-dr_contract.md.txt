# 域随机化 Contract

Manager-Based event term 是 UniLab 唯一的 DR lifecycle。任务 provider 协议已移除，
环境不再携带 DR manager。

## 生命周期

- **Construction identity：** `env.fixed_model_variants` 物化最终 read-only
  assignment，并把 UniSim `FixedVariantPlan` 附到 `SceneCfg`。backend 在 first
  forward 与 CUDA graph capture 前完成 realization。
- **Startup：** Manager 构造完成后，backend 先 materialize，event term 再采样
  一个 full-width per-environment 表，并在 read-plan 绑定、第一次 reset 或 step
  之前复用与 reset DR 相同的 backend-owned packed reset 事务
  （`SimBackend.begin_packed_reset()`）提交。提交成功后，packed composer 将
  已提交行的 staged 字段提升为其持久 reset baseline，因此后续 selected-row
  reset 会保留未写入的 startup 值。Startup 是显式 opt-in，用连续分布覆盖换取
  steady-state reset 路径中的 full-model derived-constant refresh。
- **Reset：** event term 通过 Entity binding 写入 `ResetStateTransaction`；
  transaction 将全部 staged 写入组合进 backend-owned packed 事务，并恰好提交
  一次。
- **Interval：** event term 通过公开 `SimBackend` contract 使用 backend-owned
  interval plan。

## 能力边界

Backend 差异是显式 capability，不是 task-side 分支：

- `DomainRandomizationCapabilities.supported_reset_terms`
- `supported_interval_terms`
- fixed-variant layout
- per-environment playback 支持
- `TensorLifecycleCapabilities.packed_reset` 与 `SimBackend.begin_packed_reset()`：
  Manager-Based reset 与 startup DR 生命周期的必需 capability；缺失声明会在 env
  构造时 fail closed。

未声明支持的请求 term 会携带 backend 与 term 名称 fail closed。Manager code 不
import MuJoCo 或 mjbatch，也不访问 backend model/pool。

## Reset Payload 与默认值

`ResetRandomizationPayload` 是 curated NumPy plan，首维为 selected row count。
支持项包括 body mass/COM/inertia family、gravity、geometry friction/size/solver
参数、joint damping/armature/friction 与 actuator gains。geometry bounds 等
派生字段归 backend 所有，caller 不能独立提交。

冷路径 binding 时，`ResetStateTransaction` 向 UniSim 请求
`SimBackend.get_reset_term_default(term)`。返回表格是权威默认值，只有两种布局：

- canonical model table，例如 `body_mass` 的 `(nbody,)`；
- per-environment fixed-variant table，例如 `(num_envs, nbody)`。

对 selected reset subset，packed composer 持有的 baseline 是权威来源。只写部分
model columns 时，只有这些列会被 stage；未写列保留 composer 的 per-env
baseline，并由 backend-owned composer 在唯一一次提交时合并。缺失能力、未支持
term、非 floating 表、非法 tail、首维不是 `num_envs` 的 per-env 表均 fail
closed。该边界移除了 UniLab 侧为 inertia 默认值重新 compile MuJoCo XML 的路径。

## Interval Terms

Interval plan 基于 term descriptor：`IntervalRandomizationPlan.ops` 携带来自
`unisim.dr.interval` 的 `IntervalTermOp`。内置 payload contract 由
`IntervalTermOp.validate` 强制；未知 backend-owned custom term 传给该
backend handler table。Ops 与 plans 保持 stdlib/NumPy 数据，可跨 spawn collector
pickle。

## 仓库证据

- Manager lifecycle：`src/unilab/managers/event_manager.py`
- Reset transaction：`src/unilab/base/reset_state.py`
- Entity bindings：`src/unilab/base/entity.py`
- Task-owned fixed variants：`src/unilab/base/variants.py`
- Backend contract/capability types：`unisim.backend.base`、`unisim.dr.types`
- ADR：{doc}`ADR-0010 Fixed Model Variant Ownership Boundary </adr/ADR-0010-fixed-model-variant-ownership-boundary>`
