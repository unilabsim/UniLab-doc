---
orphan: true
---

# ADR-0017 FlashSAC Learner 侧自定义 Actor 注入与 Checkpoint 语义契约

语言: 简体中文

- Status: Proposed
- Date: 2026-10-16
- Owners: RL / Runtime maintainers
- Supersedes: None
- Superseded by: None

## Context

下游算法包(驱动 issue:unilabsim/unilab_rl#30,SONIC 的 UniversalToken actor)需要替换
FlashSAC 的被优化 actor(多 encoder/decoder 结构、actor 自带的 auxiliary loss、非默认
action transform),同时复用 unilab.rl 的 FlashSAC learner、优化器、DP 梯度同步与
off-policy runtime。现有扩展点(ADR-0016 保留的 `OffPolicyActorAdapter` registry 与
`OffPolicyRuntime.runtime_resolver`)服务于"整个 algo_type 都被替换"的自定义 runtime:
runtime adapter 只接管 collector/warmup 的推理钩子,`FlashSACLearner` 仍直接构造
`FlashSACActor`,仅注册 adapter 无法一致地替换被优化的 learner actor,也无法把 actor 的
辅助标量损失接入现有 optimizer/DDP 梯度路径,更不会在 checkpoint 中记录 actor 的构造与
observation/action 语义。

## Decision

为 FlashSAC 建立 **learner 侧自定义 actor 注入契约**,owner 为
`src/unilab/rl/algos/flash_sac/actor_spec.py`:

1. **注入点**:`FlashSACLearner(actor_config=...)`,配置来自 owner Hydra YAML 的
   `algo.algo_params.actor_config = {builder: "pkg.module:callable", kwargs: {...}}`。
   builder 以 `importlib.import_module` 按 dotted path 解析(spawn 安全:任何重建 actor
   的进程重新 import 同一模块;`kwargs` 必须 JSON 可序列化),以
   `builder(obs_dim=..., action_dim=..., device=..., **kwargs)` 调用。任务/网络结构/
   损失公式归下游包所有;注入契约与生命周期归 unilab.rl。unilab.rl 不得 import 任何
   下游任务/模型代码。
2. **同一实例**:优化与推理共用 `learner.actor`(runtime 经
   `sample_offpolicy_actions` → `actor.explore` 做 learner 侧推理),不引入第二份 policy
   拷贝。现有 `OffPolicyActorAdapter` 路径(自定义 algo_type)保持不变,两条扩展路线
   互不干扰。
3. **actor 契约**:`forward(obs, training) -> (actions, info)`(`info["log_prob"]`
   必须是**变换后** action 的 log-prob,含 action transform 的 log|det J| 修正);
   `training=True` 时 `info["aux_losses"]` 可选携带 `dict[str, 0-dim Tensor]` 的
   actor 自带、已加权的 auxiliary loss;`get_mean_and_std` / `explore(dones,
   deterministic)` / `as_export_module` / `normalize_parameters` /
   `inference_startup_memory_categories` 为必需方法;`action_transform_id` 为非空
   字符串属性。构造时 fail-closed 校验(冒烟 forward/explore/export,之后恢复模型与
   RNG 状态)。`explore(deterministic=True)` 与 `as_export_module()` 必须输出
   transform(mean),保证 exploration、training、deterministic inference 与 ONNX
   export 的 action transform 与 log-prob 语义一致。
4. **auxiliary loss**:learner 把 `aux_losses` 各分量加进 actor objective,梯度走既有
   `actor_optimizer` 与 DP gradient sync;诊断按 `Loss/aux_<name>` 上报(metric schema
   新增 open family `Loss/aux_`);aux 名称集合在训练过程中必须稳定,否则 fail-closed。
5. **checkpoint 语义**:learner state dict 新增 `actor_semantics` 条目
   (`schema_version`/`kind`/`builder`/`kwargs`/`obs_dim`/`action_dim`/
   `action_transform`),`load_state_dict`(训练 resume)与 play 加载
   (`load_play_actor`)在加载权重前逐项校验,不匹配即拒绝。无该条目的旧 checkpoint
   只允许加载到内置默认 actor。
6. **CUDA graph**:自定义 actor 与 whole-cycle CUDA graph 不兼容,请求
   `use_whole_cycle_cuda_graph`(含 NVIDIA 上的默认自动开启)时 fail-closed 报错,
   不静默回退;内置 actor 行为不变。`actor_normalize_parameters=false` 仅对内置 actor
   有意义,与自定义 actor 组合时报错。

## Stable Contracts

1. `actor_config` 的 builder 为 dotted path、kwargs JSON 可序列化;spawn 子进程可通过
   同一 spec 重建 actor(`tests/rl/algos/test_flash_sac_custom_actor.py` 的 spawn 测试)。
2. auxiliary loss 梯度到达 actor 参数、`Loss/aux_<name>` 进入 canonical metric schema;
   aux 名称集合漂移直接报错。
3. checkpoint `actor_semantics` 在 resume 与 play 两条加载路径上 fail-closed 校验;
   默认 actor 的 checkpoint 行为与旧格式(无语义条目)保持兼容。
4. 自定义 action range 下,exploration/training/deterministic/export 四处 action 与
   log-prob 语义一致(变量替换公式有行为测试)。
5. 自定义 actor + `use_whole_cycle_cuda_graph` 显式报错;默认 actor 的 compile/graph
   路径不变(既有 CUDA graph 测试为回归证据)。

## Alternatives Considered

- **沿用 `OffPolicyActorAdapter` 注册自定义 algo_type**:adapter 只覆盖推理/构建钩子,
  learner 内被优化的 actor、`Loss/actor` 目标与 checkpoint 语义仍指向内置
  `FlashSACActor`,无法表达"同一个 FlashSAC learner 换 actor 并带 aux loss",故拒绝。
- **`learner_cls` 整体替换(`OffPolicyRuntime.runtime_resolver`)**:下游需复制整个
  learner 才能改 actor,违背复用现有 learner/runtime 的目标,故拒绝。
- **Hydra `_target_` instantiate**:与现有 `runtime_resolver` 的 dotted-path 约定不一致,
  且会把实例化语义隐式化;dotted-path builder 更显式、易做 spawn 安全与 checkpoint
  校验,故拒绝。

## Consequences

- 下游包必须让 builder 模块在所有训练进程可 import,并让 actor 满足契约方法集;
  违反即构造期报错而非训练期漂移。
- 自定义 actor 放弃 whole-cycle CUDA graph 吞吐;需要该优化时必须在内置 actor 路径
  验证后另行扩展契约。
- `actor_semantics` 成为 checkpoint schema 的一部分;变更其中任何字段的语义都视为
  checkpoint 兼容性问题,需要评审。
- 不允许在 unilab.rl 内出现任何下游(SONIC)import;任务专属生命周期仍归下游 owner 包。

## Evidence In Repo

- `src/unilab/rl/algos/flash_sac/actor_spec.py`(契约、构建、校验、checkpoint 语义)
- `src/unilab/rl/algos/flash_sac/learner.py`(注入点、aux loss 集成、语义持久化)
- `src/unilab/rl/algos/common/actor_factory.py`、`src/unilab/rl/algos/flash_sac/double_buffer.py`(构建接线)
- `src/unilab/conf/flashsac/config.yaml`(`algo.algo_params.actor_config`)
- `src/unilab/visualization/interactive_playback.py`(play/export 加载校验)
- `src/unilab/rl/logging/metric_schema.py`(`Loss/aux_` family)
- `tests/rl/algos/test_flash_sac_custom_actor.py`、`tests/rl/algos/flash_sac_actor_fixtures.py`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`ADR-0016 RL Reintegration Into unilab.rl </adr/ADR-0016-rl-reintegration-into-unilab-rl>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
- Driving issue: https://github.com/unilabsim/unilab_rl/issues/30
