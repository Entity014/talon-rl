# Update Functional-Effect Audit Verdict

Status: **FROZEN — NO CLEAN UPDATE DISCRIMINATOR**

Date: 2026-09-24

## Question

Does the finite policy update itself show a distinctive parameter- or function-space effect that separates next-checkpoint robust semantic collapse from retained/non-robust updates?

## Corpus

Independent v18 control-only paths:

    source-PASS transitions      22
    robust semantic collapse     10
    retained / non-robust        12

A fixed, visitation-independent probe set was collected once from the frozen base V2-B policy:

    4 reset suites
    8 fixed phases
    8 environments
    256 observations

The same probe states and the same T/A/O/S-heavy plus center preferences were used for every source and target checkpoint.

## Parameter displacement

Raw actor parameter step magnitude is effectively identical between groups:

    median ||Delta theta||
      robust collapse   0.023506093
      retained          0.023506092

    AUC                 0.396

Relative parameter norm is also non-discriminative:

    AUC                 0.304

Thus robust collapse is not explained by a larger Euclidean parameter step.

Layer-group displacement shares are also similar. Median shares:

    actor body
      robust   56.4%
      retained 57.2%

    action head
      robust   42.8%
      retained 41.8%

    preference FiLM
      robust   0.865%
      retained 0.783%

No parameter-group displacement pattern provides a clear separation.

## Fixed-state action-function displacement

Collapse-producing updates alter the fixed-state action function somewhat more:

    mean action displacement
      robust median   0.0602
      retained median 0.0564
      AUC             0.663

    max preference action displacement
      robust median   0.0604
      retained median 0.0566
      AUC             0.671

Both directions are consistent across all three seeds, but neither reaches the frozen AUC >=.75 discriminator threshold.

## Heavy-vs-center action-response preservation

Preference-conditioned action response changes are modestly larger in collapse updates:

    response-change mean AUC       0.638
    response-rotation mean AUC     0.663
    response-rotation max AUC      0.671

Mean response rotation is seed-consistent, but only 6/10 robust events exceed the retained median, below the frozen 8/10 requirement.

Target-axis response rotation itself is only slightly separated:

    robust median   0.00153
    retained median 0.00138

Therefore collapse is not cleanly characterized as a large rotation of the relevant heavy-vs-center action-response vector on fixed states.

## Preference Jacobian change

The strongest function-space near-signal is change in action sensitivity to preference:

    relative Jacobian change
      robust median   0.0587
      retained median 0.0530
      AUC             0.704
      seed direction  3/3 consistent

    Jacobian rotation
      robust median   0.00171
      retained median 0.00138
      AUC             0.696
      seed direction  3/3 consistent

These effects are directionally reproducible, but remain below the frozen AUC >=.75 criterion and only 7/10 robust events exceed the retained median.

Thus preference-Jacobian disturbance is suggestive but not a validated collapse discriminator.

## Common versus preference-specific action displacement

Most update-induced action displacement is common across preferences:

    common displacement AUC      0.663
    preference-specific AUC      0.646
    specific-fraction AUC        0.554

Median preference-specific fraction:

    robust collapse   0.0407
    retained          0.0399

This is nearly identical. Collapse-producing updates do not show evidence of a qualitatively larger preference-specific fraction of functional displacement.

## Frozen strong-discriminator gate

No metric satisfies all required conditions:
- >=8/10 robust events above retained median;
- >=0.5 retained-IQR median effect;
- ROC AUC >=.75;
- seed-consistent direction.

Therefore:

> **NO CLEAN UPDATE DISCRIMINATOR**

No function-space family and no parameter-space magnitude metric passed the predeclared gate.

## Interpretation

The audit rules out a simple explanation in which robust collapse is caused by unusually large policy steps.

It also does not support a sharply distinct preference-specific functional rewrite.

Instead, collapse-producing updates show only moderate tendencies toward:
- larger fixed-state action displacement;
- larger heavy-vs-center response rotation;
- larger preference-Jacobian change.

The strongest of these, relative Jacobian change, reaches AUC 0.704 and is directionally consistent across all three seeds, but remains insufficient to identify collapse reliably.

The appropriate conclusion is:

> **Robust semantic collapse is produced by updates whose function-space effects overlap substantially with retained updates. The failure is therefore not captured by parameter step magnitude, global fixed-state action displacement, heavy-vs-center response rotation, preference-Jacobian change, or preference-specific displacement fraction alone.**

This suggests that the decisive effect may depend on the interaction between the update and the *visited closed-loop state distribution*, rather than on a static fixed-state functional metric.

That last possibility is a hypothesis, not yet a validated mechanism.

## Decision

No trust-region, Jacobian, action-response, or preference-specific update regularizer is authorized.

No new training method is opened from this audit.

## Thesis wording

> An update-level audit found no clean finite-step discriminator between robust semantic collapse and retained competence. Actor parameter displacement was virtually identical across outcomes, excluding simple step-magnitude explanations. Collapse-producing updates showed moderately larger fixed-state action displacement and preference-Jacobian change, with the latter reaching AUC 0.704 and a consistent direction across all three seeds, but no parameter- or function-space metric met the predeclared discrimination gate. Robust collapse therefore appears to depend on effects not captured by static fixed-state update magnitude or preference sensitivity alone, consistent with an interaction between the update and subsequent closed-loop visitation.
