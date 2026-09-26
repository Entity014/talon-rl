# Update × Visitation Interaction Audit Contract

Status: **PREDECLARED — READ-ONLY, NO TRAINING / NO METHOD DESIGN**

Date: 2026-09-24

## Question

Does robust semantic collapse emerge from an interaction between the finite policy update and the closed-loop state distribution visited after that update, even when neither the update nor visitation shift alone is sufficient?

## Primary corpus

Use only the independent v18 source-PASS transitions:

    seeds 983001, 984001, 985001
    eligible transitions = 22
    robust semantic collapse = 10
    retained / non-robust = 12

Labels are frozen from `prospective_slope_validation_report.json`.
Boundary-sensitive target failures are not positive cases.

## 2×2 design

For every transition and semantic axis i:

    F(source policy, source states) = Sπ / Ss
    F(target policy, source states) = Tπ / Ss
    F(source policy, target states) = Sπ / St
    F(target policy, target states) = Tπ / St

This separates:
- policy-update effect on a fixed source distribution;
- visitation-distribution effect on a fixed source policy;
- their finite interaction.

## State-distribution collection

For each unique checkpoint and axis needed by the 22 transitions:
- execute that checkpoint policy under the corresponding axis-heavy preference;
- reset suites: 840001, 840002, 840003, 840004;
- 8 environments;
- 64 rollout steps;
- cache policy observation vectors before each action.

No center-policy rollout is used to define visitation. The state distribution is the actual heavy-preference closed-loop distribution for the semantic axis under study.

## Fixed phase partition

All 64 steps are retained and phase-resolved as:

    Q1 = 0..15
    Q2 = 16..31
    Q3 = 32..47
    Q4 = 48..63

No post-hoc phase selection is allowed.

## Functional quantities F

On each state set, evaluate source and target policies at:
- axis-heavy preference `w_i=.70`;
- center preference `.25,.25,.25,.25`.

For each policy/state-set/phase compute:

### A. Heavy-vs-center action-response magnitude

    R(s) = a(s,w_i-heavy) - a(s,w_center)

    F_resp_norm = RMS ||R(s)||_2

### B. Heavy-vs-center action-response direction

Relative to the source-policy/source-state response template for that transition and phase:

    F_resp_cos = cosine(R_current, R_source/source)

For the baseline cell Sπ/Ss this equals 1 by definition.

### C. Preference Jacobian magnitude

At center preference:

    J_aw(s) = partial a / partial w

    F_J_norm = RMS ||J_aw(s)||_F

### D. Preference Jacobian direction

Relative to the source-policy/source-state Jacobian template:

    F_J_cos = cosine(vec(J_current), vec(J_source/source))

## Interaction terms

For scalar magnitude metrics where higher means larger response:

    I_F = F(Tπ,St) - F(Tπ,Ss) - F(Sπ,St) + F(Sπ,Ss)

For preservation cosine metrics, define destruction metric:

    D = 1 - cosine

and use the same difference-in-differences:

    I_D = D(Tπ,St) - D(Tπ,Ss) - D(Sπ,St) + D(Sπ,Ss)

Positive `I_D` means target policy + target visitation destroys preference-response structure more than the additive effects of update and visitation alone.

Primary interaction metrics:

    I_resp_rotation_Q1..Q4
    I_J_rotation_Q1..Q4

Secondary:

    I_resp_norm_Q1..Q4
    I_J_norm_Q1..Q4

## Phase-growth summaries

For primary interaction rotations:

    growth_resp = I_resp_rotation_Q4 - I_resp_rotation_Q1
    growth_J    = I_J_rotation_Q4 - I_J_rotation_Q1

Also report monotonic-quarter count:

    number of adjacent increases among Q1->Q2, Q2->Q3, Q3->Q4

## Control comparisons

Primary comparison:

    10 robust-collapse transitions
    vs
    12 retained/non-robust transitions

Secondary source-margin-matched comparison uses the same deterministic rule as v19:
1. same axis when possible;
2. nearest source G_sem;
3. no reuse until unique controls are exhausted.

## Frozen strong-interaction gate

The interaction hypothesis is considered supported only if at least one primary family (`response rotation` or `Jacobian rotation`) satisfies all:

1. Q4 interaction AUC for robust collapse >= .75;
2. robust median Q4 interaction > retained median by >= 0.5 retained IQR;
3. at least 8/10 robust collapses have Q4 interaction above the retained median;
4. direction is consistent across all 3 seeds when both classes occur;
5. interaction growth Q4-Q1 AUC >= .70;
6. the corresponding non-interaction update-only metric from v19 has AUC at least .05 lower than the Q4 interaction AUC.

### UPDATE × VISITATION INTERACTION SUPPORTED
At least one primary family passes all six conditions.

### CLOSED-LOOP AMPLIFICATION WITHOUT CLEAN INTERACTION
Q4 interaction is directionally larger in collapse and phase growth is present, but the strong gate fails.

### NO INTERACTION EVIDENCE
Primary interaction metrics show no meaningful separation or no late-phase amplification.

## Decision scope

This audit does not authorize:
- visitation regularization;
- Jacobian penalties;
- state-distribution matching;
- trust-region methods;
- replay/retention;
- any new training run.

Any intervention requires a separate prospective or causal test.
