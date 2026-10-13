# ADR-0231 Integration-only K=1 MJWarp collector CUDA graph contract

语言: English

- Status: Proposed / integration-only
- Date: 2026-10-13
- Owners: UniLab runtime, UniSim MJWarp backend, uni_rl off-policy collector
- Roadmap: #2307
- Scope: `integration/issue-2307-k1-graph-local-deps` and sibling integration branches only
- Supersedes: None
- Superseded by: None

## Context

The K=1 off-policy collector has a strict transition dependency:
`observation[t] -> learner action[t] -> env.step(action[t]) -> observation[t+1]`.
Phase A profiling on the canonical MJWarp / FlashSAC / G1 motion workload found addressable collector overhead of 19.54% in independent-context mode and 32.71% under MPS. The dominant blockers are manager update and episode reset dispatch, hot validation reductions, dynamic reset-row compaction, and reset-path temporary allocation. MJWarp already owns fixed-shape physics step/reset graphs, but that does not cover Manager-based observations, rewards, termination, events, commands, packed reset, or transition publication.

This is not a production commitment, no release artifact is published, and promotion to `main` remains an owner decision outside #2307.

## Decision

Define an integration-only public capture boundary negotiated through `SimBackend` capabilities. The candidate consumes one preallocated action input of shape `[num_envs, action_dim]` and produces fixed-shape observation, reward, terminated, truncated, terminal-observation, and transition-publication carriers. The candidate is exactly K=1; policy inference and learner updates are outside capture.

### Public capability and preflight

`SimBackend` exposes a public capture candidate capability and preflight contract before any environment graph is constructed. Preflight requires fixed shapes/devices, declared stable buffers, supported Warp/Torch stream semantics, allocator readiness, RNG ownership, and capability support. Unsupported capabilities fail closed; UniLab never probes or branches on backend-private attributes.

### Buffer and stream ownership

- UniLab policy/inference IPC owns action publication and signals action readiness to the collector graph stream.
- The collector graph owns in-graph action/control reads and all fixed output carriers.
- Warp physics and Torch manager kernels execute on the declared graph stream with explicit events at stream ownership transitions.
- Replay ingress owns fixed transition slots and consumes a graph-completion event; it may overlap only after that event and must preserve contiguous tick order.
- Learner sampling and policy inference never access in-flight carriers.
- Independent-context and MPS execution use the same ordering contract; MPS changes execution sharing, not semantics.

### Lifecycle and invalidation

The lifecycle is: allocate persistent carriers, eager warmup, preflight, capture, replay, completion publication, invalidation, fallback/fail-closed. Graph state, contract version, disable reason, fallback count, replay ordering counters, and stream/RNG evidence enter the run manifest. Capture is invalidated on model/data/view/capacity/policy-I/O change, backend graph invalidation, or stream/event failure.

### Validation and semantics

Hot finite/shape checks become device-resident fixed packs. A failed device verdict fails graph completion; exact host diagnostics run at the failure boundary. Reward logs, episode metrics, termination diagnostics, reset counts, and other semantic publications move to interval/completion boundaries without deletion. Final observation, terminal transition ordering, and post-reset observation semantics remain identical to eager execution.

Torch RNG state is consumed in a deterministic capture-safe order. RNG state/offset is versioned; replay or invalidation resumes from the same recorded state and never silently advances consumption.

### Fallback policy

Malformed manifests or unsupported capabilities fail closed before capture. Runtime graph failure permits explicit eager fallback only when no transition was published for the failed replay and the manifest records reason, counts, and deterministic state rollback. Otherwise execution fails closed.

### K=1 boundary

K>1, pre-committed action sequences, stale/replay inference, and speculative transitions are outside this contract. FlashSAC `n_step=1` is TD bootstrapping, not action-block K.

## Consequences

- Graph-safe cleanup must land in owner modules before MVP; no backend-private probing is permitted.
- The contract is testable independently of performance acceptance.
- Integration branches may implement the bounded canonical workload only after Gate B static and semantic audits pass.
- Successful Gate D support is integration-only until owners promote it; failing performance remains a valid no-go outcome.

- Backend capability negotiation remains public through `unisim.backend.base.SimBackend`; backend-private probing is prohibited.
- Manager-Based training semantics, reset masks, terminal observations, transition ordering, semantic logs, and episode statistics remain eager-equivalent.
- Policy inference, learner updates, K>1 action blocks, pre-committed action sequences, and stale/replay inference remain outside capture.
- Manifests fail closed and expose graph version/state/reason/fallback counters.

## Alternatives Considered

- **Expand MJWarp physics graphs only:** rejected because Phase A shows Manager update/reset and publication overhead outside the existing physics graph.
- **Capture the learner update cycle too:** rejected because policy/model updates require outside-graph ownership and would violate the bounded collector roadmap.
- **No graph route:** remains a valid Phase B/C/D no-go if cleanup, contract, semantic, or performance gates fail.

## Evidence In Repo

- Roadmap evidence: issue #2307 Phase A aggregate
- Manager runtime: `src/unilab/envs/manager_based_rl_env.py`
- Reset transaction: `src/unilab/base/reset_state.py`
- Motion owner: `src/unilab/tasks/motion_tracking/common/manager_terms.py`
- MJWarp public backend: `../unisim/src/unisim/backend/mjwarp/backend.py`
- Collector owner: `../unilab_rl/src/uni_rl/offpolicy/worker.py`

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0012 Sole Tensor Manager And Scoped Backends </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`
- {doc}`ADR-0013 CUDA MPS Single-Rank Execution Sharing </adr/ADR-0013-cuda-mps-single-rank-execution-sharing>`
