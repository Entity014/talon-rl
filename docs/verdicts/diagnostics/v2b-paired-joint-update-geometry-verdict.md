# V2-B Paired Joint-Update Geometry Verdict

Status: FROZEN — STATIC LAMBDA=1 JOINT GRADIENT DOES NOT SUPPRESS ANGULAR; MULTI-UPDATE PATH / VISITATION FEEDBACK REMAINS

Date: 2026-09-24

## Design

Frozen V2-B model_75 checkpoint. For each heavy preference T/A/O/S and matched seed, the exact same rollout batch, rewards, values, actions, and old log-probs were used to recompute GAE with lambda=0.95 and lambda=1.0. No optimizer steps were performed.

Measured at all_actor, shared_body, direct preference input, embedding, FiLM, and actor head:
- per-objective gradient norms
- weighted contribution shares
- pairwise objective-gradient cosine
- combined-gradient cosine to heavy objective
- override ratio of non-heavy vs heavy weighted components

## Heavy-objective result

Lambda=1.0 does not suppress the heavy objective in the static combined gradient.

| Preference | all-actor cos(combined,heavy) lambda=.95 -> 1.0 | override lambda=.95 -> 1.0 |
|---|---|---|
| T | 0.877 -> 0.902 | 0.506 -> 0.492 |
| A | 0.991 -> 0.993 | 0.182 -> 0.172 |
| O | 0.993 -> 0.994 | 0.141 -> 0.139 |
| S | 0.979 -> 0.986 | 0.309 -> 0.277 |

The same qualitative result holds across shared body, direct preference columns, embedding, FiLM, and actor head.

Therefore the Angular collapse seen after full lambda=1.0 training is not explained by a one-step static joint gradient that already points away from Angular under A-heavy preference.

## Gradient scale

Lambda=1.0 increases objective-gradient magnitudes substantially on the same batch:
- Tracking: ~1.69x
- Angular: ~1.58-1.60x
- Orientation: ~1.51-1.52x
- Smoothness: ~1.65x

Combined update norm grows ~1.54-1.65x depending on preference/layer.

Thus lambda=1 changes not only direction fidelity but effective actor step scale under a fixed optimizer learning rate.

## Conflict topology

Across every heavy-preference batch, lambda=1 changes pairwise geometry consistently:
- T-A cosine becomes ~0.09 more negative
- T-O becomes ~0.08 more negative
- T-S becomes ~0.05 more negative
- A-O becomes ~0.09-0.10 more positive
- A-S becomes ~0.05 more positive
- O-S becomes ~0.09-0.10 more positive

Interpretation:
> lambda=1 strengthens a T-versus-(A/O/S) antagonistic structure while making A/O/S more mutually aligned.

This is a real static geometry change, but it does not directly explain why Angular alone loses endpoint semantics, because A-heavy combined alignment remains extremely high and slightly improves.

## Causal status

- Per-objective MC32 fidelity at lambda=1: established.
- Static heavy-objective suppression of Angular: rejected.
- Static override of A-heavy update by other objectives: rejected.
- Gradient magnitude amplification at lambda=1: supported.
- Pairwise conflict-topology shift: supported.
- Angular behavioral collapse after repeated training: remains unexplained by one-step geometry.

## Decision

The next gate should isolate repeated optimization from visitation feedback.

Run a fixed-batch multi-update path audit on cloned models from the same checkpoint:
- same frozen batch throughout
- exact same optimizer/PPO rule
- lambda=.95 vs 1.0
- no environment recollection
- measure per-step objective gradient shares/cosines and fixed-probe preference-conditioned action behavior.

If Angular degrades on the repeated fixed batch, the cause lies in parameter-space optimization-path effects / effective step scale.
If it remains stable until batches are recollected, visitation feedback is the primary remaining mechanism.