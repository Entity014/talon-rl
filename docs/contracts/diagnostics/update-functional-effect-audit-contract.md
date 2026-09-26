# Update Functional-Effect Audit Contract

Status: **PREDECLARED — READ-ONLY, NO TRAINING / NO METHOD DESIGN**

Date: 2026-09-24

## Question

What about the finite policy update itself distinguishes a next-checkpoint **robust semantic collapse** from a retained/non-robust update?

This audit intentionally moves away from source-policy scalar precursors. The unit of analysis is the source->target policy update.

## Primary corpus

Use only the independent prospective-slope validation paths:

    training seeds 983001, 984001, 985001

Eligible transitions are exactly the 22 source-PASS transitions already frozen in v18:

    robust-collapse positives = 10
    retained/non-robust       = 12

Labels are taken from the frozen v18 robust-collapse rule. Boundary-sensitive target failures are not positive cases.

## Frozen policy checkpoints

For each transition, source and target checkpoints are the already-generated V2-B control-only models under:
- Foundation V2
- GAE lambda=.95
- unchanged weighted MORL PPO
- no retention or auxiliary intervention

## Fixed functional probe set

To isolate update-induced functional change from changed visitation, collect one independent fixed observation probe set once from the frozen base V2-B checkpoint using center preference only:

    reset seeds: 850001, 850002, 850003, 850004
    64 steps per suite
    8 envs

Subsample deterministic phases:

    t = {0, 8, 16, 24, 32, 40, 48, 56}

Expected probe observations:

    4 suites x 8 phases x 8 envs = 256 states

The same observations are used for every source and target policy.

## Preference set on fixed probes

Evaluate exactly:

    T-heavy = [.7,.1,.1,.1]
    A-heavy = [.1,.7,.1,.1]
    O-heavy = [.1,.1,.7,.1]
    S-heavy = [.1,.1,.1,.7]
    Center  = [.25,.25,.25,.25]

No additional preference points are added after seeing results.

## Metric family 1 — parameter displacement

Actor parameter vector includes the same trainable policy-side parameters used in prior actor-update audits:
- actor_* parameters
- preference_embedding*
- preference_film*
- log_std

For every transition report:

    ||Delta theta||_2
    ||Delta theta||_2 / ||theta_source||_2

and layer-group displacement shares for:
- actor trunk/body
- preference embedding
- preference FiLM
- action head / log_std when identifiable

Parameter norm is descriptive; no claim is made that Euclidean parameter distance equals functional distance.

## Metric family 2 — fixed-state action-function displacement

For every preference p in {T,A,O,S,C} on the 256 fixed probes:

    D_action(p) = RMS || a_target(s,p) - a_source(s,p) ||_2

Aggregate:

    D_action_mean = mean_p D_action(p)
    D_action_max  = max_p D_action(p)

Also report preference heterogeneity:

    CV_action_change = std_p D_action(p) / (mean_p D_action(p)+eps)

## Metric family 3 — heavy-vs-center response preservation

For each axis i define the preference-conditioned action response at a policy checkpoint:

    R_i(s) = a(s,w_i-heavy) - a(s,w_center)

For source->target update report per axis:

    response-change RMS:
        ||R_i^target - R_i^source||

    response cosine:
        cos(R_i^source, R_i^target)

    response rotation:
        1 - cosine

Aggregate across axes:

    response_change_mean
    response_rotation_mean
    response_rotation_max

This family asks whether the update preserves the functional meaning of increasing a preference coordinate relative to center.

## Metric family 4 — preference Jacobian change

At the same fixed probe states and center preference, compute action Jacobian:

    J_aw = partial a / partial w

using autograd at source and target checkpoints.

Report:

    Jacobian relative change:
        ||J_target - J_source||_F / (||J_source||_F + eps)

    Jacobian cosine

    Jacobian singular values / effective rank

    change in column norms for T/A/O/S preference directions

Primary aggregate:

    J_change_rel
    J_rotation = 1 - cosine(vec(J_source), vec(J_target))

## Metric family 5 — common vs preference-specific functional displacement

For each state, define update-induced action change:

    Delta a_p(s) = a_target(s,p) - a_source(s,p)

Common component:

    Delta a_common(s) = mean_p Delta a_p(s)

Preference-specific residual:

    Delta a_spec_p(s) = Delta a_p(s) - Delta a_common(s)

Report RMS magnitudes:

    D_common
    D_specific
    specific_fraction = D_specific / (D_common + D_specific + eps)

This tests whether collapse-producing updates disproportionately alter preference-specific behavior rather than merely shifting the policy in a common direction.

## Matched comparison

Primary comparison is all 10 robust-collapse updates versus all 12 retained/non-robust updates.

Secondary matched comparison pairs each robust update to the retained/non-robust transition with:
1. same axis when possible;
2. nearest source continuous G_sem;
3. no reuse until unique controls are exhausted.

Because samples are modest, report effect sizes and rank statistics rather than relying on significance claims.

## Frozen decision rules

A metric family is a **strong update discriminator** only if all hold:

1. direction is consistent in at least 8/10 robust collapses relative to the median retained value;
2. robust median differs from retained median by at least 50% of the retained interquartile range (IQR; denominator floored at eps);
3. ROC AUC for robust collapse >= .75 in the expected risk direction;
4. direction is consistent across all 3 training seeds when both classes occur.

For metrics where larger means more destructive change, risk score is the metric itself.
For cosine/preservation metrics, risk score is negated preservation or explicit rotation quantity.

### FUNCTIONAL UPDATE EFFECT SUPPORTED
At least one function-space family (action displacement, response rotation, Jacobian change, or preference-specific fraction) satisfies the strong-discriminator rule, while raw parameter norm alone does not explain the same separation.

### PARAMETER-MAGNITUDE EFFECT ONLY
Only parameter displacement satisfies the rule; function-space metrics do not.

### NO CLEAN UPDATE DISCRIMINATOR
No metric family satisfies the rule.

## Scope

This audit does not authorize:
- trust-region penalties;
- Jacobian regularization;
- action-response retention;
- preference-specific update constraints;
- optimizer changes;
- any new training run.

Any intervention requires a separate prospective or causal gate after this diagnostic.
