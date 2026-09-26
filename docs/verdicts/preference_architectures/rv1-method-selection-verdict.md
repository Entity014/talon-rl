# RV1 Method-Selection Verdict

Status: FROZEN — RV1-C SEMANTIC FAIL ON VALIDATED FOUNDATION; V2 METHOD BRANCH AUTHORIZED

Date: 2026-09-23

## Final causal resolution

The final critic blocker was traced to a checkpoint freshness mismatch, not a failure of the actor-training credit path.

Training order in RV1:

    collect rollout with actor_t
    recollect/refit critic head on current-policy support
    compute vector advantages with the refreshed head
    actor update -> actor_{t+1}
    save actor_{t+1} together with head_t

Therefore the saved checkpoint can contain a one-update actor/head mismatch even though each actor update used a freshly refit head for actor_t.

## Actor-frozen head-refresh audit

Checkpoint: runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt

A. Current-policy selected-12 refit:
- first-H32 EV mean: 0.5471
- first-H32 negative fraction: 0.0000
- second-H32 EV mean: 0.1195
- second-H32 negative fraction: 0.2375
- second-H32 mean abs bias: 0.0336
- Tracking second-H32 EV: +0.0872
- Orientation second-H32 EV: +0.0986
- Late critic gate: PASS

B. Current-policy expanded-support refit:
- first-H32 EV mean: 0.6091
- first-H32 negative fraction: 0.0000
- second-H32 EV mean: 0.1884
- second-H32 negative fraction: 0.0875
- second-H32 mean abs bias: 0.0348
- Tracking second-H32 EV: +0.1034
- Orientation second-H32 EV: +0.1810
- Late critic gate: PASS with larger margin

Baseline saved head:
- second-H32 EV mean: -0.0085
- second-H32 negative fraction: 0.4500
- Orientation EV: -0.0927
- FAIL

Interpretation:
- A passing is sufficient to show that current-policy head refresh closes the formal late critic freshness gate.
- B's larger margin shows support breadth still improves robustness/generalization.
- Weighting redesign and nonlinear head/body redesign remain unsupported by current evidence.

## RV1 semantic result

The actor is unchanged by the head-refresh audit. Therefore the previously measured critic-independent RV1-C behavior result remains valid:

- endpoint semantic response: FAIL overall
- continuum monotonicity: below predeclared threshold
- endpoint-between fraction: below predeclared threshold
- survival: preserved
- tracking collateral: bounded
- preference sensitivity: present

Thus direct conditioning is used by the actor but does not produce the required continuous 4D semantic behavior under the repaired pipeline tested here.

## Decision

RV1-C = FAIL on a validated foundation.

Supported scope:

> Minimal direct preference conditioning is insufficient to produce the required semantic preference response under the validated repaired RV1 training pipeline and evaluation contract.

This is not a universal impossibility claim about all direct-conditioning policies.

## Authorization state

    RV1-A                     PASS / FROZEN
    RV1-B                     PASS / FROZEN
    critic temporal persistence repaired
    critic phase coverage diagnosed
    final head freshness gate PASS after current-policy refresh
    RV1-C semantic response   FAIL
    RV1-D                     NOT AUTHORIZED
    V2 method branch          AUTHORIZED
    C44 variance intervention OFF

The next V2 experiment must preserve the validated foundation and introduce only the minimum additional preference-representation/modulation mechanism needed to test why direct conditioning was insufficient.