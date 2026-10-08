---
orphan: true
---

# ADR-0013 CUDA MPS Single-Rank Execution Sharing

语言: 简体中文

- Status: Proposed
- Date: 2026-10-04
- Owners: UniLab training runtime maintainers
- Supersedes: None
- Superseded by: None

## Context

off-policy 训练已经要求每个 rank 的 learner 与 collector 使用同一张 rank-local GPU，
但二者仍是独立 CUDA context，由驱动按多 context 粗粒度分时仲裁。Discussion #1800
的参考主机证据显示，显式启用 NVIDIA CUDA MPS 可以让这两个 context 共享 GPU 并重叠
kernel，从而改善端到端吞吐；其证据范围是单主机、单 rank、MJWarp。

现有配置没有表达该 host execution mode，用户只能在部署层隐式设置环境变量。这既不能
fail closed，也没有可审计的 runtime evidence。

## Decision

- 新增共享 off-policy owner 设置
  `training.cuda_process_sharing: null | mps`，默认 `null` 保持现状。
- 初始有效范围限定为 Linux、NVIDIA CUDA、`training.sim_backend=mjwarp`、单主机、
  `world_size=1` 的 SAC/FastSAC/FlashSAC 共享路径。
- `mps` 是显式请求：所有前置条件在 env probe、learner construction 与 collector
  spawn 之前验证，失败时给出第一个未满足条件和启动既有 daemon 的命令，不 fallback。
- rank-local 物理一致性按 GPU UUID 判断，而不是 CUDA ordinal。
- UniLab 只验证既有 MPS control daemon 与 control socket/FIFO，不 start/stop/repair
  daemon，不引入 `auto`、SM percentage 或 OS nice/affinity 配置。
- `run_config.json` 记录配置值；runtime manifest v1 增加 producer 诊断字段
  `cuda_process_sharing`，记录 configured/effective、devices、UUID、control pipe、
  server PID 与 validated 状态。该字段先保持 v1 诊断扩展，不提升为稳定公共字段。
- 多 GPU DP 继续由 #2063 单独决策；本 ADR 的 per-rank 参数与 evidence shape 保留
  加法扩展空间。

## Stable Contracts

- Owner 配置：三个 off-policy 主配置中的 `training.cuda_process_sharing`。
- Fail-closed probe：`src/unilab/training/cuda_process_sharing.py`。
- Builder wiring：`src/unilab/scripts/train_offpolicy.py` 在构造 env/runner 前 probe，
  并把 evidence 合并进 runner 的 runtime manifest。
- 测试与文档：probe fake、owner default、builder fail-closed、manifest evidence、
  英文/中文生产指南。

## Alternatives Considered

- 继续要求用户只在 shell 设置 MPS 环境变量：无法验证拓扑，也没有结构化证据。
- 由 UniLab 启动/停止 daemon：把共享 host 服务生命周期混入训练进程，且无法安全服务
  并发/多用户运行。
- CPU priority/affinity：Discussion #1800 的测量显示其不解决跨进程 CUDA context
  arbitration，且引入未证实的公共配置复杂度。

## Consequences

- 显式 `mps` 请求失败时不会有默认多 context fallback；这避免静默性能和拓扑漂移。
- MPS daemon、pipe/log 目录与多用户隔离仍是部署者责任。
- 非 MJWarp backend 和 DP 请求会失败，不形成支持声明。
- `env_steps_per_sync` 仍是训练语义 owner 设置，不由 execution-sharing mode 修改。

## Evidence In Repo

- `src/unilab/training/cuda_process_sharing.py`
- `src/unilab/scripts/train_offpolicy.py`
- `src/unilab/conf/sac/config.yaml`
- `src/unilab/conf/flashsac/config.yaml`
- `tests/training/test_cuda_process_sharing.py`
- `tests/algos/test_offpolicy_double_buffer_runner.py`
- `docs/sphinx/source/en/2-user_guide/1-training/7-tensor_runtime_production.md`
- `docs/sphinx/source/zh_CN/2-user_guide/1-training/7-tensor_runtime_production.md`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
- Discussion #1800
- Issue #2063
