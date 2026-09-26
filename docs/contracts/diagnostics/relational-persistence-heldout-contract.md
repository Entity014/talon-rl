# Relational-Persistence Held-Out Validation Contract

Status: **PREDECLARED — HELD-OUT, MEASUREMENT ONLY, NO REWARD REDESIGN**

Date: 2026-09-24

## Purpose

The discovery attribution set suggested that semantic competence tracks the temporal persistence of matched heavy-versus-center behavioral advantage substantially better than the heavy-policy scalar return alone.

This gate tests whether that relationship reproduces on policy paths and training seeds that were never used in the discovery attribution.

No descriptor, window, threshold, or model is fit after observing held-out results.

## Strict held-out policy set

Primary validation uses only the CONTROL arms of the existing bounded-Delta-a multi-seed experiment:

- train seed 980001: base -> control u1 -> ... -> u8
- train seed 981001: base -> control u1 -> ... -> u8
- train seed 982001: base -> control u1 -> ... -> u8

Base checkpoint:
- `runs/v2b_adam_continuous-2026-09-24/model_75.pt`

The three control paths are ordinary weighted-PPO continuations and are independent of the discovery paths used in the original trajectory-attribution audit.

Primary held-out sample size:
- 3 independent training seeds
- 8 finite transitions per seed
- 24 policy transitions
- 96 axis-transition samples

The action-response rehearsal arms are excluded from the primary test to avoid mixing a retention treatment into the mechanism-validation sample.

## Evaluation protocol

For every unique checkpoint, rerun the exact frozen endpoint protocol:
- 8 environments
- 64 steps
- 4 matched suites
- reset seeds 840001..840004
- T/A/O/S-heavy plus center preference
- same V2-B architecture and objective normalization
- same physical semantic proxies

No optimizer step is allowed.

## Frozen semantic targets

For each axis i at each checkpoint:

- objective correctness fraction: fraction of 4 suites where heavy objective > matched center objective
- physical correctness fraction: fraction of 4 suites where heavy physical proxy < matched center physical proxy
- semantic score:

      S_i = 0.5 * (objective_correct_fraction + physical_correct_fraction)

- PASS_i: unchanged frozen endpoint criterion

For each consecutive transition compute:
- Delta objective-correctness fraction
- Delta physical-correctness fraction
- Delta semantic score
- PASS -> FAIL / FAIL -> PASS

This target is deliberately suite-directional rather than the mean heavy-center margin, avoiding the tautology identified in the discovery family comparison.

## Frozen descriptors

All windows and definitions are frozen from the discovery audit.

### Baseline
1. `J_heavy`: 64-step mean normalized objective signal under heavy preference.

### Absolute-heavy temporal control
2. `heavy_obj_late`: heavy objective mean over steps 43..63.
3. `heavy_phys_late`: negative heavy physical mean over steps 43..63, so higher is better.

### Relational mean control
4. `obj_adv_mean`: 64-step mean heavy-minus-center objective advantage.
5. `phys_adv_mean`: 64-step mean center-minus-heavy physical advantage.

### Relational temporal candidates
6. `obj_adv_late`
7. `phys_adv_late`
8. `obj_adv_late_minus_early`
9. `phys_adv_late_minus_early`
10. `obj_adv_slope`
11. `phys_adv_slope`
12. `obj_adv_final16`
13. `phys_adv_final16`
14. `obj_adv_worst16`
15. `phys_adv_worst16`
16. `obj_adv_positive_fraction`
17. `phys_adv_positive_fraction`

Definitions exactly match the discovery audit:
- early = steps 0..20
- late = steps 43..63
- final16 = steps 48..63
- worst16 = minimum rolling 16-step relational mean
- slope = least-squares slope across all 64 steps
- positive fraction uses zero tolerance 1e-10

No alternative window length is permitted in this gate.

## Primary continuous comparisons

Objective-side descriptors are compared with Delta objective-correctness fraction.
Physical-side descriptors are compared with Delta physical-correctness fraction.
`J_heavy` is compared with Delta objective-correctness fraction.

Report for every descriptor:
- Spearman correlation
- Pearson correlation
- sign agreement
- per-training-seed Spearman
- per-axis Spearman

## Mechanism controls

The key comparison is hierarchical:

    heavy-only total
        vs
    heavy-only late
        vs
    heavy-center mean
        vs
    heavy-center temporal persistence

This distinguishes:
- temporal information alone;
- relational information alone;
- the combination of relation + temporal persistence.

## PASS -> FAIL primary check

For every PASS -> FAIL event report whether each frozen descriptor deteriorates.

Primary event descriptors:
- J_heavy
- heavy_obj_late / heavy_phys_late
- obj_adv_mean / phys_adv_mean
- obj_adv_late / phys_adv_late
- late-minus-early relational advantage
- final16 relational advantage
- worst16 relational advantage
- positive-advantage fraction

PASS -> FAIL evidence is primary because forgetting is the target phenomenon.

## Predeclared held-out validation gate

The relational-temporal mechanism is considered **HELD-OUT VALIDATED** only if all hold:

1. **Relational temporal reproduction**
   - at least 4 of the 6 paired relational-temporal groups
     (`late`, `late-minus-early`, `slope`, `final16`, `worst16`, `positive_fraction`)
     have absolute Spearman >= 0.50 on both their objective and physical targets.

2. **Improvement over heavy-only total**
   - median absolute Spearman across the 12 relational-temporal descriptors exceeds `|rho(J_heavy)|` by >= 0.15.

3. **Relation adds information beyond temporal-only heavy statistics**
   - both `obj_adv_late` and `phys_adv_late` have absolute Spearman at least 0.10 greater than their corresponding heavy-only late controls.

4. **Temporal persistence adds information beyond relational mean**
   - at least one objective temporal descriptor and at least one physical temporal descriptor have absolute Spearman at least 0.10 greater than the corresponding relational-mean descriptor.

5. **Seed consistency**
   - for each of `obj_adv_late`, `phys_adv_late`, `obj_adv_late_minus_early`, and `phys_adv_late_minus_early`, the correlation sign agrees with the overall held-out sign in all 3 training seeds.

6. **PASS -> FAIL consistency**
   - at least 75% of held-out PASS -> FAIL events show deterioration in both late objective relational advantage and late physical relational advantage.
   - the same fraction for `J_heavy` must be at least 0.15 lower than the late-relational fraction, demonstrating added forgetting sensitivity.

If there are fewer than 4 PASS -> FAIL events, criterion 6 is marked underpowered and the gate cannot authorize reward redesign even if continuous criteria pass.

## Decision

### HELD-OUT VALIDATED
Relational temporal persistence is validated as a generalizing missing-information mechanism. This authorizes a separate, minimal reward/objective-design branch, but does not authorize full training directly.

### PARTIAL / INCONCLUSIVE
Some relationships reproduce but one or more mechanism controls fail, or PASS -> FAIL events are underpowered. Reward redesign remains blocked.

### NOT REPRODUCED
Relational temporal descriptors fail to outperform the heavy-only and/or relational-mean controls. Reward redesign remains blocked and the discovery attribution is treated as non-generalizing.

No reward coefficient, objective term, architecture, optimizer, lambda, replay, or semantic threshold is changed in this gate.
