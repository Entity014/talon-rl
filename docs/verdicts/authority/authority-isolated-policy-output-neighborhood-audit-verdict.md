# Authority-Isolated Policy-Output Neighborhood Robustness Verdict

Status: **FROZEN — GATE FAIL; STOP RESIDUAL-MECHANISM MINING**
Date: 2026-09-25

## Question

Is the repaired controller sitting in an unusually narrow neighborhood of stable endogenous action sequences, such that small generic perturbations of its own policy output reproduce delayed closed-loop failures?

## Frozen comparison

Actors:

    u50             robust reference
    u75             problematic late authority-isolated actor
    repaired-u10    projected + active tail-descent partial repair

Suites:

    suite2 seed 840003
    suite3 seed 840004

Preferences:

    O / S / C

Vulnerability rates were computed only on lanes that survive baseline under all three actors for the corresponding suite/preference cell.

## Perturbation definition

Perturbation was applied in pre-tanh output space:

    z' = z + delta
    a' = tanh(z')

Because actors have very different tanh saturation, delta was rescaled per timestep so that the realized RMS action deviation approximately matched a common action-equivalent budget.

Budgets:

    epsilon_a = .01
    epsilon_a = .02
    epsilon_a = .04
    epsilon_a = .08

Perturbations were smooth over t0..15.

Generic morphology-only spatial bases:
- global/common
- left-right
- front-rear
- diagonal
- hip-vs-distal
- hip group
- thigh group
- calf group

Temporal bases:
- constant
- ramp
- half-sine
- full-sine
- two-lobe

No preference label, lane identity, or failure-specific joint identity was used.

## Search/validation separation

To avoid tuning perturbations independently to each test cell:

Development search used only:

    u75
    suite3
    aggregate O/S/C

The selected morphology/temporal mode was then frozen and evaluated on:

    u50 / u75 / repaired-u10
    suite2 / suite3
    O / S / C

Energy-matched smooth random perturbations were evaluated as controls.

## Baseline

u50 survives every lane in all tested suite/preference cells.

u75 reproduces its known nominal failures.

repaired-u10 reproduces its known residual failures.

Common-survivor support therefore excludes nominally failing lanes before perturbation sensitivity is computed.

## Results

### epsilon_a = .01

Development search finds no vulnerability:

    development failure score = 0

Validation aggregate new delayed failure rate:

    u50       0
    u75       0
    repaired  0
    random    0

### epsilon_a = .02

Again:

    development failure score = 0

Validation:

    u50       0
    u75       0
    repaired  0
    random    0

No narrow-neighborhood ordering appears.

### epsilon_a = .04

Development search still finds no failure on u75 suite3 O/S/C:

    development failure score = 0

The tie-selected generic mode is the global constant morphology mode.

One local validation vulnerability appears:

    u75 / suite2 / C
      new delayed failure = lane2 @ step15
      rate on common survivor support = 1/7 = .1429

The same cell:

    u50       0/7
    repaired  0/7

Aggregate across all six suite/preference cells:

    u50       0.0000
    u75       0.02381
    repaired  0.0000
    random    0.0000

This is evidence that repaired-u10 can be less locally sensitive than u75 in at least one policy-output direction.

However:
- the signal occurs in only one of six cells;
- it does not recur across O/S/C;
- the mode was not identified as adversarial on the development support;
- there is no graded u50 < repaired < u75 ordering;
- the effect is non-monotonic with perturbation magnitude.

Therefore this is a localized sensitivity observation, not a general policy-neighborhood vulnerability mechanism.

### epsilon_a = .08

Development search again finds no vulnerability.

Validation returns to:

    u50       0
    u75       0
    repaired  0
    random    0

Thus the isolated .04 effect is not a monotonic small-neighborhood fragility signature.

## Random control

Across all predeclared budgets and cells:

    smooth isotropic random delayed failure rate = 0

There is therefore no evidence that generic action-output noise broadly destabilizes the controller on common-survivor support.

## Gate evaluation

Required evidence:

1. baseline-valid common survivor support:
       PASS

2. small generic policy-output perturbations reproduce delayed failures:
       WEAK / LOCAL ONLY

3. u75 materially more vulnerable than u50:
       LOCAL ONLY at epsilon=.04, suite2/C

4. repaired-u10 consistently reduces vulnerability versus u75:
       LOCAL ONLY

5. recurrence across multiple O/S/C cells:
       FAIL

6. morphology-aware perturbation stronger than random:
       NOT GENERALLY ESTABLISHED
       one isolated structured failure vs zero random, but no recurrence

7. desired robustness ordering across the validation envelope:
       FAIL

Final:

    POLICY-OUTPUT NEIGHBORHOOD ROBUSTNESS GATE = FAIL

## Interpretation

The data do not support the hypothesis that the remaining Class-B/residual failures arise from a generally narrow local neighborhood of stable action sequences.

Small generic perturbations with realized action RMS up to approximately .08 usually leave common-survivor trajectories stable.

The single epsilon=.04 u75 suite2/C failure shows that isolated local sensitivity exists, and that the repaired policy may remove some of it, but the phenomenon is:
- sparse,
- non-monotonic,
- non-recurrent across preferences,
- and not recovered by the development adversarial search.

This is insufficient to justify a generic local policy-smoothing or action-noise robustness method.

## Stop-rule decision

The predeclared stop condition is met.

Do not continue residual-mechanism search via:
- larger policy-output perturbation budgets;
- joint-specific perturbation search;
- preference-specific perturbations;
- lane-specific perturbations;
- more temporal-basis fishing;
- local smoothing objectives;
- action-noise robustness training derived from these residuals.

Class-B/residual mechanism mining is now closed.

## Recommended strategic redirect

The next robustness step should be broad engineering hardening rather than a mechanism-specific residual patch:

    broaden training/reset distribution
    + expand validation envelope
    + preserve authority-isolated actor contract
    + preserve repaired wide critic
    + preserve projected/headroom repair

Then re-evaluate the global survival gate.

If broader distribution hardening raises semantic-suite robustness sufficiently without damaging authority, return to AI-C2 and then AI-H2.

Current authorization:

    scalar basin repair                    CLOSED
    trajectory-representation repair       CLOSED
    external-wrench adversarial training   NOT AUTHORIZED
    policy-output smoothing training       NOT AUTHORIZED
    residual mechanism mining              CLOSED
    broad distribution hardening           AUTHORIZED AS NEXT DIRECTION
    AI-C2 / AI-H2                          BLOCKED pending robustness
