# V2-B Horizon × Visitation Credit Audit

Status: FROZEN — TEMPORAL-HORIZON EFFECT DOMINANT; VISITATION EFFECT SECONDARY; V2-C REMAINS OFF

Date: 2026-09-23

## Clean design

The final audit uses common random numbers and nested continuations.
For each frozen state, a single 32-step continuation is sampled. MC_8, MC_16, MC_32, and GAE are all derived from that same continuation, so horizon is the only treatment variable.

State sources:
- center visitation
- O-heavy visitation

Phases:
- t8
- t20
- t36
- t52

All continuations evaluate O-heavy credit from the frozen state.

## GAE alignment versus MC horizon

| Source | Phase | GAE↔MC8 | GAE↔MC16 | GAE↔MC32 | MC8↔MC32 |
|---|---:|---:|---:|---:|---:|
| Center | t8 | 0.196 | 0.269 | 0.665 | 0.253 |
| Center | t20 | 0.289 | 0.402 | 0.605 | 0.581 |
| Center | t36 | 0.007 | 0.175 | 0.476 | 0.039 |
| Center | t52 | 0.274 | 0.396 | 0.608 | 0.289 |
| O-heavy | t8 | 0.155 | 0.292 | 0.680 | 0.218 |
| O-heavy | t20 | 0.019 | 0.293 | 0.581 | 0.361 |
| O-heavy | t36 | 0.153 | 0.287 | 0.543 | 0.106 |
| O-heavy | t52 | 0.225 | 0.445 | 0.529 | 0.259 |

Across both state sources and all phases, GAE aligns much better with the longest MC horizon than with the shortest horizon.

MC_8 and MC_32 themselves often have low cosine, especially at t36. Therefore the Orientation policy gradient geometry is strongly horizon-dependent even when state and random continuation are held fixed.

## Visitation-source effect

At matched phase and horizon, the difference between O-heavy-source and center-source GAE↔MC cosine is usually smaller and less systematic than the horizon effect.

Largest notable source difference occurs at t20 / H8 (O-heavy lower by ~0.27), but this difference shrinks strongly at H32 (~-0.02). Other source effects change sign across phase/horizon.

Thus visitation source matters in some regimes but does not provide a consistent dominant explanation.

## Interpretation

The previous hypothesis that credit drift is mainly caused by visiting a different state distribution is too strong.

The clean CRN audit supports a stronger statement:

> Orientation credit geometry is intrinsically continuation-horizon dependent. Short-horizon MC directions can differ substantially from long-horizon MC directions at the same frozen state, while GAE is consistently more aligned with the long-horizon MC direction.

This implies that temporal propagation through dynamics changes the semantic gradient geometry before any architecture issue is considered.

The closed-loop O-heavy failure should therefore be interpreted primarily as a temporal credit / dynamics-horizon realization problem, with visitation effects as a secondary interaction rather than the main driver.

## Causal status

- Conditioning authority: established.
- Action realization: established.
- Cross-objective override: rejected as primary.
- Objective semantic sign error: rejected.
- One-step local FD comparator: invalid.
- Universal late-state intrinsic GAE collapse: not supported.
- Horizon-dependent MC geometry: strongly supported.
- Visitation-source effect: present but secondary / inconsistent.
- Dynamics-mediated temporal credit geometry: primary remaining explanation.

## Decision

V2-C remains OFF.

The next intervention, if any, should target temporal credit structure rather than conditioning capacity. Before changing PPO/GAE, a minimal read-only follow-up should compare alternative advantage horizons / lambda values against the same long-horizon MC target on frozen trajectories. Architectural escalation is not justified by the current evidence.