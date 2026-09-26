# Separated Anchor-Policy Causal Control Contract

Status: **PREDECLARED — POLICY-FAMILY HYPOTHESIS TEST**

Date: 2026-09-24

## Question

Does semantic competence become more stable when the principal preferences are trained as **independent policies**, removing shared-actor overwrite while preserving the validated objective/foundation semantics?

This is a causal control for the shared-policy bottleneck hypothesis, not a deployment architecture.

## Frozen foundation

- Base initialization: `runs/v2b_adam_continuous-2026-09-24/model_75.pt`
- Actor architecture inside each copy: V2-B single-site FiLM
- Foundation V2 reward/objective definitions
- repaired transformed-action PPO semantics
- GAE lambda = .95
- same rollout horizon, environment count, actor parameter set, and nominal finite actor-step norm used in the validated v18 control paths
- same semantic evaluation reset suites: 840001..840004
- same 64-step evaluation horizon
- no replay, retention, ranking, temporal, context, trust-region, Jacobian, manifold, or architecture intervention

## Independent policy family

Clone the exact same base checkpoint into five independent policies:

    T  [.70,.10,.10,.10]
    A  [.10,.70,.10,.10]
    O  [.10,.10,.70,.10]
    S  [.10,.10,.10,.70]
    C  [.25,.25,.25,.25]

Each policy is updated only using its own fixed preference. There is no parameter sharing after initialization.

## New training seeds

Exactly:

    986001
    987001
    988001

Each seed trains all five policies for 8 finite updates.

For update k and family member p, use a deterministic batch seed derived from the training seed, update index, and fixed policy index. The five policies do not share optimizer state or parameters.

## Fixed-preference actor update

For policy p with fixed preference w_p:

1. collect the same H-step current-policy rollout as the v18 control;
2. compute normalized 4D objective advantages using lambda=.95;
3. form the same clipped weighted PPO actor loss using fixed w_p;
4. take the normalized actor-gradient direction;
5. apply exactly the same nominal finite actor step norm used by the v18 control path.

Critic/value parameters remain unchanged, matching the v18 finite-update causal-control path. This isolates actor sharing only.

## Family-level semantic evaluation

At checkpoint u=0..8, evaluate each independent policy on its own fixed preference over the four frozen reset suites.

For each semantic axis i compare the axis specialist Pi against the center specialist PC on the **same reset suite**:

    objective margin = J_i(Pi) - J_i(PC)
    physical margin  = Phys_i(PC) - Phys_i(Pi)

where higher positive margin is semantically correct.

Axis i PASS iff:

    objective-positive suites >= 3/4
    physical-positive suites  >= 3/4
    minimum specialist survival >= .95
    minimum center survival     >= .95

Continuous family gate margin is also reported using the frozen v15 normalization scales and second-smallest suite margin.

## Stability metrics

For each axis / training seed report:

- first PASS checkpoint;
- final PASS;
- retained-PASS fraction after first PASS;
- maximum continuous gate-margin forgetting after best-so-far;
- PASS->FAIL event count;
- number of consecutive PASS checkpoints after first acquisition.

Family-level accumulation at checkpoint u:

    N_pass(u) = number of T/A/O/S axes passing simultaneously

Report maximum and final N_pass and whether N_pass decreases after reaching a new maximum.

## Primary causal gate

### SEPARATION SUPPORTS SHARED-POLICY BOTTLENECK
Require all:

1. At least 3/4 semantic anchors acquire PASS in every training seed.
2. Median retained-PASS fraction after first acquisition >= .75 across acquired axis-seed pairs.
3. Final simultaneous family competence `N_pass >= 3` in every seed.
4. Median robust continuous-margin forgetting is smaller than the frozen shared-policy control corpus by at least 25% when measured on comparable acquired axes.
5. PASS->FAIL rate after acquisition is <=25% across anchor-policy transitions.
6. No seed shows a repeated winner-rotation pattern in which `N_pass` falls by >=2 after reaching its maximum.

### SEPARATED ANCHORS STILL SEMANTICALLY UNSTABLE
If >=3/4 anchors acquire competence but retention criteria 2/3/5 fail, then removing parameter sharing is not sufficient to solve semantic instability.

### ANCHOR OBJECTIVES THEMSELVES INSUFFICIENT
If fewer than 3/4 anchors acquire PASS in at least two of three seeds, the fixed-preference objective/task formulation is insufficient even without shared-policy overwrite.

### INCONCLUSIVE
Use only if acquisition is too sparse or safety failures prevent interpretation.

## Decision scope

- If shared-policy bottleneck is supported: authorize a separate **explicit continuous policy-family** design branch (substantial Hyper-MORL / PSL-style parameter generation), not training yet.
- If separated anchors remain unstable: do not build a full hypernetwork as a forgetting fix.
- If anchor objectives are insufficient: return to objective/task formulation rather than policy-sharing architecture.

This control does not itself authorize deployment or a continuous policy-family architecture.
