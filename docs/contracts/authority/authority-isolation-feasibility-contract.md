# Authority-Isolation Feasibility Audit Contract

Status: **PREDECLARED — SUPERVISED FUNCTION TRANSFER ONLY; NO RL TRAINING**
Date: 2026-09-25

## Question

Can the frozen V2-B actor function be transferred with low error into an actor whose **shared trunk is state-only** and whose **only actor-side preference path is a policy-family generator**?

This audit tests architectural transferability only. It does not test semantic improvement.

## Teacher

Frozen teacher:

    V2-B + Foundation V2
    runs/v2b_adam_continuous-2026-09-24/model_75.pt

Teacher actor preference dependence may use:
- direct preference concatenation;
- preference embedding;
- FiLM.

## Student authority-isolated actor

Actor requirements:

    observation -> state-only shared trunk -> h
    preference w -> family generator -> generated conditional policy block
    h + generated block -> action mean

The actor shared trunk must never receive w.
No direct w concatenation, preference embedding, or FiLM actor path is allowed.

Critic is outside this feasibility question and remains unchanged / unused.

## Student capacity

Freeze one architecture before fitting:

    state-only trunk: 48 -> 128 -> 128 -> 128
    family hidden: 128
    family bases: K=8
    coefficient MLP: 4 -> 32 -> 8

The generated family block uses full-rank learned bases for:

    128 -> 128
    128 -> 12

plus biases.

No objective-labelled modules are allowed.

## Initialization

State-only trunk copies the observation columns and downstream trunk parameters from the frozen V2-B actor wherever shapes permit.

The family block may be initialized independently.

No claim of exact function preservation is made before supervised transfer.

## Fixed state corpus

Use only already-cached observations; no simulator rerun.

### Transfer-fit states

Use source-visitation state caches from training seeds:

    983001
    984001

Pool all unique checkpoint-axis caches from these seeds, then select a deterministic evenly spaced subset after concatenation:

    8192 states

### Held-out states

Use source-visitation state caches from seed:

    985001

select deterministic evenly spaced subset:

    4096 states

Additionally evaluate the independent fixed probe set:

    runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz
    256 states

No state from seed 985001 or the fixed-probe set enters fitting.

## Preference corpus

Preference samples are frozen before fitting.

### Fit preferences

- T/A/O/S/C anchors;
- 59 interior simplex samples from Dirichlet(1,1,1,1), NumPy RNG seed 26092501.

Total:

    64 fit preferences

### Held-out preferences

64 independent interior simplex samples from Dirichlet(1,1,1,1), RNG seed 26092502.

No held-out preference enters fitting.

## Supervised target

Transfer the teacher's **pre-tanh deterministic action mean**:

    mu_teacher(s,w)

Student minimizes mean-squared error:

    L_transfer = E ||mu_student(s,w)-mu_teacher(s,w)||^2

No reward, rollout return, PPO objective, semantic label, collapse label, or critic target is used.

## Optimization budget

This is function fitting, not RL.

Freeze:

    optimizer = Adam
    learning rate = 3e-4
    batch size = 2048 state-preference pairs
    steps = 5000
    weight decay = 0

Each minibatch independently samples fit states and fit preferences using RNG seed 26092503.

Do not early-stop on held-out metrics.
Save checkpoints at steps:

    0, 500, 1000, 2500, 5000

## Evaluation metrics

Evaluate teacher vs student pre-tanh means and tanh actions on four splits:

1. fit states × fit preferences;
2. held-out states × fit preferences;
3. fit states × held-out preferences;
4. held-out states × held-out preferences.

Also evaluate fixed probe states × held-out preferences.

Report:

    RMSE pre-tanh mean
    mean L2 action error
    p95 L2 action error
    max absolute action-coordinate error
    pairwise preference-separation relative error
    preference-Jacobian relative Frobenius error at center
    preference-Jacobian cosine at center

## Frozen feasibility gate

### FUNCTION-PRESERVING-ISH TRANSFER FEASIBLE

Require all on **held-out states × held-out preferences**:

1. pre-tanh RMSE <= 0.02;
2. mean tanh-action L2 error <= 0.03;
3. p95 tanh-action L2 error <= 0.06;
4. max absolute action-coordinate error <= 0.10;
5. pairwise preference-separation relative error <= 10%;
6. center preference-Jacobian relative Frobenius error <= 15%;
7. center preference-Jacobian cosine >= 0.95.

And on independent fixed probes × held-out preferences:

8. mean tanh-action L2 error <= 0.04;
9. preference-separation relative error <= 15%.

Generalization guard:

10. held-out/fit pre-tanh RMSE ratio <= 1.5.

### TRANSFER APPROXIMATE BUT NOT FUNCTION-PRESERVING

If held-out mean action error <=0.08 and Jacobian cosine >=0.85, but the full gate fails.

This would require any future authority-isolated branch to be treated as a **non-function-preserving architecture replacement**, not a continuation of V2-PF.

### AUTHORITY ISOLATION TRANSFER NOT FEASIBLE

If held-out mean action error >0.08 or Jacobian cosine <0.85.

## Decision scope

This audit authorizes no RL training by itself.

- If function-preserving-ish transfer is feasible: a separately predeclared H0/H1 authority-isolated RL branch may be considered.
- If approximate only: decide explicitly whether a non-function-preserving replacement experiment is worth the confound.
- If not feasible: close authority-isolation architecture under this transfer route.

No transfer threshold or architecture dimension may be changed after observing results.
