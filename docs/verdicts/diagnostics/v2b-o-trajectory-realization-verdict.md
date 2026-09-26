# V2-B O-Heavy Trajectory Realization Audit

Status: FROZEN — ACTION REALIZATION PRESENT; CLOSED-LOOP SEMANTICS BECOME STATE/TIME DEPENDENT; V2-C REMAINS OFF

Date: 2026-09-23

## Design

Frozen V2-B checkpoint, no optimizer steps.
Matched O-heavy versus center stochastic rollouts:
- 4 matched reset suites x 8 environments = 32 paired trajectories
- 64 steps
- identical reset seeds within each O/center pair
- identical policy-noise seeds within each O/center pair

Logged per timestep:
- tilt
- angular velocity xy
- vertical velocity
- joint state norms
- action/action-rate
- FiLM gamma/beta
- normalized Orientation reward
- Orientation MC return
- Orientation GAE advantage
- score-gradient contribution

## Action realization

O-heavy changes actions immediately. Mean paired action-vector distance O-heavy vs center:

| Window | action-vector distance | tilt delta (O-C, deg) | O-reward delta | GAE-MC score cosine (O) |
|---|---:|---:|---:|---:|
| t00-07 | 0.0846 | -0.0145 | +0.000026 | +0.703 |
| t08-15 | 0.1402 | -0.0140 | +0.000077 | +0.821 |
| t16-23 | 0.3691 | +0.0844 | -0.000616 | +0.785 |
| t24-31 | 0.6779 | -0.1354 | +0.000839 | +0.547 |
| t32-39 | 1.0635 | +0.0965 | -0.001101 | +0.212 |
| t40-47 | 1.2207 | +0.2577 | -0.001462 | +0.040 |
| t48-55 | 1.4351 | -0.0282 | +0.000851 | -0.291 |
| t56-63 | 1.5831 | -0.3071 | +0.004271 | -0.462 |

Action-vector distance is already >0.05 at t=0 and grows strongly as the trajectories separate. Therefore the Orientation failure is not explained by preference authority failing to reach the action space.

FiLM modulation itself is preference-dependent from the first step. Because this single-site FiLM generator depends only on w, the O-center gamma/beta norm offsets are constant across time; the growing action difference comes from closed-loop state divergence through the modulated actor.

## Early semantic realization

During approximately the first 16 steps, O-heavy is on average slightly flatter than center and Orientation reward is slightly better. This shows that the preference-conditioned action change can initially realize the intended physical semantic direction.

A sustained tilt-worse interval first appears around t=18. The trajectory subsequently alternates between better and worse tilt rather than undergoing one irreversible sign flip.

Thus the failure is not well described as a simple local action sign error. It is a closed-loop, state-dependent trajectory effect.

## Temporal credit behavior

The strongest time-local signal is the decay of Orientation GAE-score alignment with the long-horizon MC Orientation score direction:
- early: approximately 0.70-0.82
- t24-31: approximately 0.55
- t32-39: approximately 0.21
- t40-47: approximately 0.04
- t48-55: approximately -0.29
- t56-63: approximately -0.46

This decay occurs after substantial state/action divergence has developed. By late trajectory phases the local GAE-weighted score direction can oppose the MC-return score direction.

## Causal interpretation

Supported:
- O-heavy preference changes action immediately and substantially.
- O-heavy can initially improve tilt/reward.
- closed-loop state divergence grows over time.
- Orientation behavior becomes oscillatory/non-monotonic across trajectory phases.
- GAE-vs-MC Orientation credit alignment degrades strongly with time/state divergence and becomes negative late.

Not established:
- that GAE temporal drift alone causes the endpoint failure.
- that nonlinear dynamics coupling alone causes the endpoint failure.
- that adding conditioning capacity would fix either mechanism.

Current best-supported locus:
> The remaining Orientation failure occurs after correct preference authority reaches the action space, in the state-dependent closed-loop realization of credit over time. Temporal GAE/MC credit drift and coupled dynamics are both implicated, with temporal credit drift providing the clearest measured phase-dependent signal.

## Decision

V2-C remains OFF.

The next experiment should remain read-only and isolate the state/time dependence of Orientation credit around the divergence transition (roughly t16-t47), for example by replaying matched states from early/mid/late phases and comparing local deterministic/MC/GAE action-gradient directions at the same states. No additional FiLM sites, adapters, routing, or auxiliary losses are authorized yet.