# V2-A1 Authority / Foundation Verdict

Status: FROZEN — ACTOR AUTHORITY PASS; FOUNDATION BREADTH BLOCKED; V2-A2 NOT AUTHORIZED

Date: 2026-09-23

## What V2-A1 established

The learned preference embedding is actively used by the actor under the same repaired PPO/objective foundation:

- embedding readout norm: 0 -> 0.1692
- max embedding-parameter gradient norm: 0.00943
- median embedding-parameter gradient norm: 0.00233
- median embedding-readout gradient norm: 0.10093
- total pairwise fixed-state action distance: 0.04360
- masked-embedding pairwise action distance: 0.03687
- total preference Jacobian norm: 0.02747
- masked-embedding Jacobian norm: 0.02456
- mean embedding action authority: 0.05530
- PPO ratio invariant: PASS
- last-10 termination fraction: 0.0125

Thus V2-A1 passes the actor-side authority question: the embedding acquires causal action authority beyond the retained RV1 direct path.

## Critic foundation check

The saved checkpoint head is stale by construction and therefore is not used for the final foundation decision.

Actor-frozen current-policy refresh results:

A. selected-12 current-policy refit:
- first-H32 aggregate EV: +0.5573
- second-H32 Tracking EV: +0.2646
- second-H32 Orientation EV: +0.1054
- second-H32 aggregate negative fraction: 0.2625
- bias bounded
- formal late gate: FAIL only because negative fraction exceeds 0.25

B. expanded current-policy refit:
- first-H32 aggregate EV: +0.6420
- second-H32 aggregate EV: +0.1935
- second-H32 Tracking EV: +0.2673
- second-H32 Orientation EV: +0.1934
- second-H32 aggregate negative fraction: 0.1500
- second-H32 mean abs bias: 0.0331
- formal late gate: PASS

## Interpretation

V2-A1 does not fail because the embedding lacks authority. It fails to progress because the frozen selected-12 critic support breadth that was sufficient for repaired RV1 is marginally insufficient under the new V2-A actor visitation distribution.

Expanded current-policy support is sufficient to restore the critic freshness gate without changing the actor, critic body, PPO rule, reward, objectives, or embedding mechanism.

However, adopting expanded support only for V2 would violate the architecture-only comparison contract. Therefore V2-A2 remains blocked until the foundation contract is amended fairly for both RV1 and V2 (or another architecture-neutral support rule is predeclared and validated on both).

## Authorization

    V2-A0                         PASS / FROZEN
    V2-A1 actor authority         PASS
    V2-A1 frozen-foundation gate  BLOCKED
    V2-A2 semantic gate           NOT AUTHORIZED
    V2-B FiLM                     OFF
    adapters / routing            OFF

Next decision: either retain the original selected-12 foundation and stop V2-A here, or amend the critic-support foundation architecture-neutrally and revalidate both RV1 and V2-A under the same amended support rule before semantic comparison.