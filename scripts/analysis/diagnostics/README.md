# Diagnostic implementations

[Analysis](../README.md)

Implementation modules behind [`scripts/analysis/run_diagnostics.py`](../run_diagnostics.py).

These modules are grouped here because they are **analysis operations**, not training entry points.

## What lives here

The current diagnostics fall into a few recurring families:

### Rollout and gait traces

Examples:

- `balance_progress_trace.py`
- `command_switch_trace.py`
- `gait_joint_trace.py`
- `leg_collapse_trace.py`
- `prefall_window_analysis.py`

Use these to inspect how behavior evolves through time.

### Update / optimization diagnostics

Examples:

- `advantage_decomposition.py`
- `full_update_trace.py`
- `mean_update_decomposition.py`
- `minibatch_composition_trace.py`
- `multi_update_trace.py`
- `ppo_update_diagnostic.py`

Use these when the question is about PPO credit, gradients, minibatches, or update geometry.

### Objective and semantic probes

Examples:

- `objective_perturbation_probe.py`
- `objective_segment_audit.py`
- `progress_leak_counterfactual.py`
- `compare_gradients.py`
- `compare_valuation.py`

Use these to test whether objective semantics and policy response agree.

### Physical / actuator checks

Examples:

- `gravity_actuator_consistency.py`
- `gravity_finite_diff_check.py`
- `ground_contact_timestep_check.py`
- `instantaneous_equilibrium_check.py`
- `open_loop_step_response.py`
- `torque_authority_ablation.py`

Use these to separate policy failures from simulator/plant inconsistencies.

### Intervention and attribution

Examples:

- `hip_intervention_trace.py`
- `hip_symmetry_intervention.py`
- `hip_functional_correlation.py`
- `command_sensitivity_probe.py`
- `stochastic_diversity_probe.py`

Use these for controlled interventions and causal-attribution-style checks.

## Shared helpers

`_common.py` contains environment/setup helpers shared by several diagnostics. Keep it implementation-focused; reusable RL abstractions should move to `scripts/rl/core/`.

## Adding a diagnostic

1. Add a module in this folder with a `main()` function.
2. Reuse `_common.py` where appropriate.
3. Register a stable kebab-case command in `../run_diagnostics.py`.
4. Keep training-side behavior unchanged; diagnostics should observe or intervene deliberately, not silently alter the trainer.
