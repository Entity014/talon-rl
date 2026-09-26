# Separated Anchor-Policy Causal Control Verdict

Status: **FROZEN — SEPARATED ANCHORS STILL SEMANTICALLY UNSTABLE**

Date: 2026-09-24

## Question

Is semantic forgetting primarily caused by forcing all preferences to update one shared conditional actor?

The causal control removes this sharing completely while preserving the validated training foundation.

## Experimental treatment

Three new training seeds:

    986001
    987001
    988001

For each seed, five independent V2-B copies were initialized from the same frozen `model_75.pt`:

    T-heavy
    A-heavy
    O-heavy
    S-heavy
    center

After initialization, the five policies shared no parameters and no optimizer state.

Each policy received only its own fixed preference for 8 finite actor updates using:
- Foundation V2 objectives;
- repaired PPO action/log-prob semantics;
- GAE lambda=.95;
- the same 32-step rollout construction;
- the same normalized actor-step norm as the validated v18 controls (`0.02350609`);
- no retention, auxiliary loss, architecture change, or regularization.

## Semantic evaluation

At u0..u8, each axis specialist was compared against the independently trained center specialist on the same four reset suites.

For axis i:

    objective margin = J_i(P_i) - J_i(P_C)
    physical margin  = Phys_i(P_C) - Phys_i(P_i)

PASS requires positive objective and physical margins on at least 3/4 suites plus survival >=.95.

This tests the semantic family relation directly without requiring one policy to represent multiple preferences.

## Acquisition

Separated policies can clearly **acquire** semantic competence.

Across 12 axis-seed pairs:

    acquired PASS at least once = 11 / 12

Per seed:

    986001   4 / 4 anchors acquire
    987001   4 / 4 anchors acquire
    988001   3 / 4 anchors acquire

Thus failure cannot be reduced to an inability of fixed-preference policies to ever reach semantically correct behavior.

The only non-acquired pair is Angular in seed 988001.

## Retention

Acquisition does not remain stable.

Across the 11 acquired axis-seed pairs:

    median retained-PASS fraction after first acquisition = 0.25

Representative cases:

### seed 986001

    Tracking     first PASS u2   retained fraction 0.00
    Angular      first PASS u2   retained fraction 0.17
    Orientation  first PASS u1   retained fraction 0.14
    Smoothness   first PASS u0   retained fraction 0.25

No axis is PASS at the final checkpoint.

### seed 987001

    Tracking     first PASS u2   retained fraction 1.00
    Angular      first PASS u2   retained fraction 0.17
    Orientation  first PASS u4   retained fraction 0.00
    Smoothness   first PASS u0   retained fraction 0.375

Only Tracking remains PASS at the final checkpoint.

### seed 988001

    Tracking     first PASS u4   retained fraction 0.75
    Angular      never PASS
    Orientation  first PASS u4   retained fraction 0.50
    Smoothness   first PASS u0   retained fraction 0.625

Only Tracking remains PASS at the final checkpoint.

## Family-level accumulation

Number of simultaneously correct semantic anchors:

### seed 986001

    N_pass timeline = [1,1,2,1,1,0,0,2,0]
    maximum         = 2
    final           = 0

### seed 987001

    N_pass timeline = [1,1,2,2,2,2,1,2,1]
    maximum         = 2
    final           = 1

### seed 988001

    N_pass timeline = [1,1,1,1,3,2,1,2,1]
    maximum         = 3
    final           = 1

Therefore explicit separation does not produce monotonic semantic accumulation.

Two seeds lose at least two simultaneous competencies after reaching their running maximum:

    986001  drop = 2
    988001  drop = 2

The phenomenon is no longer literally "winner rotation inside one shared parameter vector," but the **family-level semantic relation still rotates/degrades over training**.

## PASS-to-FAIL frequency

Across all post-acquisition transitions:

    PASS->FAIL events               = 18
    post-acquisition transitions    = 67
    PASS->FAIL rate                 = 26.9%

Frozen requirement:

    <=25%

Result:

    FAIL

Even by this relatively permissive criterion, separated anchors remain unstable.

## Continuous gate-margin forgetting

Separated anchors do not improve continuous semantic forgetting relative to the independent shared-policy v18 controls.

Median maximum gate-margin forgetting after acquisition:

    separated anchors   6.887
    shared-policy v18   3.625

A 25% reduction was required to support the shared-policy bottleneck hypothesis.

Instead, separated anchors show approximately:

    +90% larger median forgetting

on this metric.

Because the experiments use different training seeds, this comparison should be interpreted as a control-level effect rather than a paired numerical estimate. However, it clearly does not support the required improvement.

## Frozen primary gate

### 1. >=3/4 anchors acquire in every seed

    PASS

### 2. median retained-PASS fraction >=.75

    FAIL: 0.25

### 3. final N_pass >=3 in every seed

    FAIL: 0 / 1 / 1

### 4. continuous-margin forgetting improves >=25%

    FAIL: forgetting is larger, not smaller

### 5. PASS->FAIL rate <=25%

    FAIL: 26.9%

### 6. no >=2 drop from maximum family competence

    FAIL in 2/3 seeds

Formal verdict:

> **SEPARATED ANCHORS STILL SEMANTICALLY UNSTABLE**

## Causal interpretation

This is an important negative result.

The experiment directly removes the mechanism that motivated a full explicit policy-family architecture:

    no shared actor parameters
    no preference competition inside one evolving parameter vector
    no cross-preference gradient overwrite between family members

Yet semantic competencies are still acquired and subsequently lost.

Therefore:

> **Shared conditional-policy parameter overwrite is not a sufficient explanation for the observed semantic instability.**

Equivalently, giving each anchor preference its own complete policy does not reliably stabilize the heavy-versus-center semantic mapping under the current objective/training formulation.

This does not prove that parameter sharing has zero effect. It shows that eliminating sharing is insufficient to solve the problem.

## Implication for continuous policy-family methods

Under the predeclared decision tree, a substantial Hyper-MORL / PSL-style policy-family branch is **not authorized as a forgetting fix**.

A larger hypernetwork may offer other benefits, but this control removes the key causal rationale for building it specifically to solve semantic retention:

    if fully independent anchor policies are themselves unstable,
    then a continuously generated family with shared hypernetwork parameters cannot be expected to solve the same failure merely by providing more parameter separation.

The next formulation-level question is therefore not "how should policy parameters be shared?" but:

> **Why does continued optimization under a fixed preference fail to preserve the semantic relation that the policy previously acquired?**

Given the preceding v15-v21 evidence, reopening low-dimensional reward/descriptor tuning is not justified. This result instead strengthens the thesis limitation that the current training objectives are not sufficient to guarantee stable semantic preference realization over successive updates, even without cross-preference actor sharing.

## Decision

- Explicit continuous Hyper-MORL / PSL family as a semantic-forgetting fix: **BLOCKED**.
- Post-hoc Pareto-family construction as a deployment alternative: **not tested here; remains a formulation-level fallback, not an authorized current branch**.
- New scalar reward / retention / architecture / regularizer search: **not authorized**.
- Final reference controller remains V2-B + Foundation V2 + lambda=.95 with no retention intervention unless a separately motivated formulation-level study is opened.

## Thesis wording

> To test whether semantic forgetting was primarily caused by parameter interference in a shared preference-conditioned actor, five independent policies were trained for the four heavy preference anchors and the center preference, with no parameter sharing after identical initialization. Semantic competence remained acquirable: 11 of 12 axis-seed pairs passed at least once. However, retention remained poor, with a median retained-PASS fraction of 0.25, final simultaneous competence of only 0–1 axes across the three seeds, and a 26.9% PASS-to-FAIL rate after acquisition. Continuous semantic-margin forgetting was also not reduced relative to the shared-policy control. Therefore eliminating cross-preference actor sharing was insufficient to stabilize semantic behavior, indicating that shared-parameter overwrite is not a sufficient explanation for the observed semantic instability.
