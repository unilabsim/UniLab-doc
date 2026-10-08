# Latency Budget

This page documents the latency controls that are visible in the repository and
the deploy-side measurements you need before hardware bring-up. Treat numeric
budgets as robot-specific measurements, not UniLab defaults.

## Latency Surfaces In Repo

| Surface | Repo evidence | What it covers |
| --- | --- | --- |
| One-step action delay | Manager action term `simulate_action_latency` declarations in task owners | Executes the previous action instead of the current action. |
| Observation history | Per-term `history_length` declarations in task owners | Per-term history for selected observation terms. |
| Obs history ordering | `ObservationManager` per-term history buffers (`tests/managers/test_observation_buffers_noise.py`) | Per-term oldest-first flatten for actor observations. |

## Action Latency

For Manager-Based tasks that enable action latency, the action manager applies
the previous action when the flag is enabled. Keep this in the selected task
owner YAML instead of adding deploy-only behavior later.

```yaml
env:
  actions:
    joint_pos:
      simulate_action_latency: true
```

## Observation Lag And History

Observation width is the sum of `dim * history_length` over the declared actor
terms, not something a hardware runtime may guess. Terms carrying
`history_length` are flattened oldest-first within each term while other terms
stay single-step.

Do not lag command/reference terms unless the training owner did so.

## Deploy-Side Measurements

Record these per policy tick in the hardware runtime:

1. `policy_input_timestamp`
2. source timestamps for each sensor or estimator channel
3. `policy_output_timestamp`
4. actuator command send timestamp
5. the action vector before and after clamp / smoothing

Compare the observation vector against a sim rollout built from the same task
owner YAML. If the measured pipeline needs filtering or buffering, encode the
matching behavior in the task owner and retrain, rather than adding it only on
the deploy side.

## Symptoms Of Mismatch

- Contact oscillation after enabling torque.
- Action saturation during the first few policy ticks.
- Velocity tracking drift even when the ONNX input width and observation layout
  match.

## See also

- {doc}`6-domain_randomization`
- {doc}`7-safety_layers`
- `src/unilab/managers/event_manager.py`
