# Prospective Relational-Slope Validation Contract

Status: **PREDECLARED — NEW CONTROL SEEDS, NO METHOD CHANGE**

Date: 2026-09-24

## Question

Does within-rollout heavy-versus-center relational slope measured at a source checkpoint prospectively identify whether currently competent behavior will undergo a robust semantic collapse at the next checkpoint, beyond source mean relation alone?

This validates diagnosis only. It does not introduce a training intervention.

## Frozen controller / training contract

- V2-B single-site FiLM actor
- Foundation V2
- GAE lambda = .95
- repaired transformed-action PPO semantics
- same weighted MORL objective
- same mixed-preference update construction
- same nominal actor step-norm contract
- 8 finite updates from the frozen `model_75.pt` base
- no replay, retention, architecture, optimizer, ranking, context, or temporal auxiliary term

## New independent training seeds

Exactly:

    983001
    984001
    985001

These seeds were not used in v17 robust-collapse mechanism analysis.

Evaluation reset suites remain frozen:

    840001, 840002, 840003, 840004

## Checkpoint measurement

At base and after every update u1..u8, evaluate for each axis i:
- heavy preference `w_i=.70`, others `.10`
- center preference `.25,.25,.25,.25`
- 4 matched reset suites
- 64 rollout steps

Store raw per-step normalized objective-i and physical metric time series.

## Source features

All predictors are measured at checkpoint t before observing checkpoint t+1.

For each axis:

### Mean relation

    M_obj  = mean_t,suite [ obj_i(heavy) - obj_i(center) ]
    M_phys = mean_t,suite [ phys_i(center) - phys_i(heavy) ]

Higher is better.

### Full-rollout relational slope

Fit ordinary least-squares slope over time index 0..63 to the suite-mean relation sequence:

    S_obj_64
    S_phys_64

More negative means relational competence erodes during rollout.

### Early-half relational slope

Same OLS slope using only steps 0..31:

    S_obj_32
    S_phys_32

This is the primary prospective feature family because it is available before the final half of the rollout.

### Continuous semantic gate margin

Using the frozen v15 discovery scales:

    G_sem = min(second-smallest normalized objective suite margin,
                second-smallest normalized physical suite margin)

Binary source PASS must match `G_sem > 0` exactly.

## Next-checkpoint robust-collapse target

A transition is eligible only when source checkpoint is PASS.

Target `Y=1` (robust semantic collapse) iff target checkpoint:
1. is semantic FAIL under the frozen gate;
2. boundary clearance >= .25;
3. source->target PASS->FAIL survives tau in {-0.25,0,+0.25};
4. exact paired reset-bootstrap PASS->FAIL probability >= .50.

Otherwise `Y=0` means retained/non-robust next-step outcome.

Boundary-sensitive failures are not positive targets.

## Frozen prospective scores

No coefficient fitting is allowed.

Per axis, discovery scales from v15 are reused to standardize objective vs physical quantities.

Define:

    MeanScore  = min(M_obj/s_obj, M_phys/s_phys)

    Slope64    = min(S_obj_64/s_obj, S_phys_64/s_phys)

    Slope32    = min(S_obj_32/s_obj, S_phys_32/s_phys)

More negative scores indicate greater collapse risk.

Risk scores used for ROC/AUC are therefore:

    R_mean    = -MeanScore
    R_slope64 = -Slope64
    R_slope32 = -Slope32

No weighting or temperature is tuned.

## Primary prospective gate

The slope hypothesis is considered prospectively validated only if all hold:

1. At least 8 eligible source-PASS transitions and at least 3 robust-collapse positives exist across the new seeds; otherwise UNDERPOWERED.
2. `AUC(R_slope32) >= 0.75`.
3. `AUC(R_slope32) >= AUC(R_mean) + 0.10`.
4. `AUC(R_slope64) >= 0.75`.
5. Median `Slope32` is more negative in future robust collapses than retained/non-robust transitions in all 3 seeds when each seed has both classes; seeds lacking both classes are reported but not counted against this criterion.
6. At a frozen risk split `Slope32 < 0`, robust-collapse rate is at least 0.25 higher than for `Slope32 >= 0`.
7. Source mean relation alone does not already achieve equivalent discrimination within 0.05 AUC of Slope32.

## Secondary diagnostics

Report:
- AUC of objective-only and physical-only slope components;
- source `G_sem` AUC;
- per-axis event counts;
- per-seed event counts;
- whether full-rollout slope materially outperforms early-half slope;
- false-high-risk rate among retained transitions.

Secondary metrics cannot rescue a failed primary gate.

## Decisions

### PROSPECTIVE SLOPE VALIDATED
Authorize a separate temporal-relational target design audit only; still no training.

### SLOPE DIAGNOSTIC ONLY
Slope distinguishes collapse retrospectively but does not prospectively outperform mean relation on independent seeds. No temporal target authorized.

### UNDERPOWERED / INCONCLUSIVE
Too few eligible or positive events. No target authorized.

No method change is authorized by this contract itself.
