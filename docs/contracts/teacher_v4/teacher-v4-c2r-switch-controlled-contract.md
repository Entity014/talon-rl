# Teacher V4 — V4-C2R Switch-Controlled Semantic Effect Contract

Status: **DRAFT — the δ-derivation rule below is fixed before the null run; the rest is frozen after the null run and before the heavy-vs-heavy test**
Branch: `v4-c2-semantic-preservation`
Date: 2026-09-27
Supersedes, for A and S interpretation, the center-comparator test in
[teacher-v4-c2-preference-effect-verdict.md](../../verdicts/teacher_v4/teacher-v4-c2-preference-effect-verdict.md),
which stays frozen as a methodological finding (switch transient).

## Question

After every branch undergoes the same kind of preference switch, and after
the switch transient has passed, does asking for objective i do better *on
i* than asking for another objective j?

    D_{i,j} = S_i(w_i⁺) − S_i(w_j⁺),   j ≠ i
    D_i     = mean over j ≠ i of D_{i,j}

S_i is the higher-is-better semantic score defined in the semantic-relation
contract. w_k⁺ is m = 4 heavy (0.70 on k, 0.10 on the others).

## Common setup

- Checkpoints: V4-C `model_300.pt`. Obs corruption off.
- Snapshots: 256 envs × env seeds {0, 1} = 512 per checkpoint, taken after
  100 warm-up steps on the center preference. The snapshot is the unit.
- Every branch is restored in place from the snapshot and **switches at t = 0
  from center to a heavy preference**. It runs 128 steps.
- Windows: **steady = steps 33–128 (primary)**; transient = steps 1–32
  (descriptive only: the cost of switching).
- Snapshots with an episode end inside 128 steps in any scored branch are
  excluded and counted.

## Step 1 — switch-matched null (run first)

Per snapshot: one burn-in branch (heavy (r+1) mod 4, discarded), then three
branches that all use the **same** heavy preference k = r, where r = env
index mod 4. Null statistics per objective i and window: mean and std of
S_i(pos2) − S_i(pos1) and S_i(pos3) − S_i(pos2), with bootstrap CIs over
snapshots. Run on G1-2 s73104 and G1-3 s73104, the same two checkpoints as
the earlier null run.

**δ-derivation rule (fixed now, before the null run):**

    δ = max(0.002, 1.2 × max |mean null|)

The max runs over both checkpoints, all four objectives, both position
differences, and the steady window. δ is then frozen and never changed
after heavy-vs-heavy results are seen.

## Step 2 — heavy-vs-heavy test (after δ is frozen)

Per snapshot: one burn-in branch (heavy (r+1) mod 4, discarded), then five
branches in the order given by a cyclic shift of [T⁺, A⁺, O⁺, S⁺, r⁺] by
(env index + env seed) mod 5. Here r⁺ repeats the heavy preference of
objective r = env index mod 4. Every objective's heavy branch appears once,
and one appears twice (the per-snapshot null). Positions are balanced by
the shift.

Per checkpoint: for each objective i, D_i in the steady window, with
one-sided 95% simultaneous bounds across the four objectives (paired
bootstrap by snapshot, max-t, B = 2000). Pairwise D_{i,j} and the transient
window are reported, not gated.

    correct: LCB > δ      wrong: UCB < −δ      flat: otherwise

Fidelity-invalid rule: the per-snapshot repeat gives mean(S_i(r⁺ second) −
S_i(r⁺ first)) in the steady window, for each i. If any |value| > δ, the
checkpoint is fidelity invalid and not interpreted.

Checkpoints: the 16 V4-C runs; primary the 10 seed-sensitivity runs.

## Interpretation (fixed now)

- For each objective, count correct / flat / wrong over the valid primary
  checkpoints, and cross-tabulate with the m = 4 endpoint result.
- T, O and S are positive controls. If none of them is correct in most valid
  checkpoints, the measurement is suspect and A is not interpreted.
- T/O/S mostly correct and A not correct → A-specific semantic limitation at
  steady state, now free of the switch confound.
- A correct → the earlier A failures (endpoints, center comparator) came
  from the center-relative comparison and the switch transient, not from
  A's own steady-state direction.
- Transient-window D is reported as the switching cost per objective.
