# Tensor Runtime Productionization

This page defines review ownership and acceptance rules for the scoped
single-GPU tensor runtime. Operational reproduction instructions live in
{doc}`../2-user_guide/1-training/7-tensor_runtime_production`; generated support
claims live in {doc}`../5-reference/5-support_matrix`.

## Contract Ownership

| Owner | Responsibility |
| --- | --- |
| UniSim | Public `SimBackend` tensor lifecycle, execution/data-plane/process profiles, packed host-bridge contracts, platform support inventory, and backend adapters. |
| UniLab task owners | Hydra owner identity, `training.sim_backend`, task semantics, parity fixtures, process-device binding, and training-entrypoint integration. |
| `uni_rl` | Off-policy collector/learner process topology, CUDA inference ring, replay ingress, IPC synchronization, runner metrics, and metric/runtime-manifest producer schemas. |
| UniLab training runtime | Soak monitoring, schema consumption, safe defaults and bounds, process-affinity integration, run artifacts, and fail-closed construction checks. |
| UniLab documentation | Reproduction, operations, schema compatibility, support-matrix interpretation, and release-transition guidance. |

Tasks and training code must consume only the public `SimBackend` interface.
They must not probe backend-private methods or turn an `unknown` platform
capability into support. Unsupported tensor lifecycles fail closed.

## Schema and Version Policy

The current namespaces are independent:

| Namespace | Version | Producer owner |
| --- | ---: | --- |
| TensorBoard / W&B metric schema | `1` | `uni_rl.logging.metric_schema` |
| Runtime process/device/IPC manifest | `1` | `uni_rl.logging.runtime_manifest_schema` |
| UniLab long-soak monitor artifact | `0.3.0` | `unilab.training.soak` |

Within a major version, producers may add optional fields with schema, tests,
and documentation. Stable fields cannot be renamed, deleted, retyped, or changed
in unit or null semantics. Consumers reject missing and unsupported versions
rather than guessing. A soak-artifact update does not change metric semantics,
and a metric-schema update does not invalidate every runtime-manifest field.

The complete compatibility contract is maintained in
{doc}`../2-user_guide/1-training/3-logging`. Do not copy that table into backend
or task documentation as a second source of truth.

## Benchmark Acceptance

There are two non-interchangeable evidence types:

1. The canonical G1 FlashSAC / MJWarp long soak is an owner-routed end-to-end
   training-lifecycle test. A current-head pass requires current schema
   versions, completed training, drained inference and replay ingress, normal
   shutdown, no residual processes, and bounded resource growth.
2. The phase-local backend probe measures synchronized backend/manager phases
   with inference IPC, replay ingestion, learner update, and production dispatch
   excluded. Its totals are diagnostic and must not be reported as end-to-end
   training throughput.

A historical soak artifact remains useful lifecycle evidence only for its
recorded commits and schema. It cannot substitute for a current-head run after
producer schema or lifecycle contracts change.

When retaining benchmark evidence, store the soak JSON, console log, run
directory, revision and package provenance, GPU identity, and checksum together.
The repository does not track machine-generated benchmark outputs.

## Support-Matrix Promotion

The generated matrix is authoritative. A passing local benchmark does not
automatically change an entrypoint evidence grade. Promotion to `Tested` or
`Benchmarked` requires a narrow maintainer-validation entry or checked-in
benchmark manifest, generator updates, focused tests, and regenerated English
and Chinese blocks.

The current FlashSAC / `g1_motion_tracking` / MJWarp production guide remains a
`Configured` owner path. It is not a default recommendation and makes no
multi-GPU, macOS, ROCm, PyPI-release, or universal-backend claim.

## Release Transition

Before advancing the published dependency versions, complete the checklist in
{doc}`5-contributing_workflow`: released UniSim tensor contracts and adapters,
released `uni_rl` runner behavior, published MJBatch support, passing focused
CPU and gated CUDA profiles on the exact transition commit, and completed M11
soak, leak, shutdown, schema, platform, and documentation acceptance.

No PyPI publication is part of M11. A release requires a separate maintainer
decision and release tag.
