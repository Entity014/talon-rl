# Teacher V4 — V4-C3 Evaluation Contract (DRAFT)

Status: **DRAFT — freeze before any V4-C3 evaluation (training runs in progress; no evaluation run)**
Branch: `v4-c2-semantic-preservation`
Training: [teacher-v4-c3-training-contract.md](teacher-v4-c3-training-contract.md)

## What changed since V4-C G1, and why

Lessons from V4-C and V4-C2, applied in advance:

1. **Critic validity uses the MC256 target** (G1-R form), not truncated H32.
   The H32 target was incompatible with a long-horizon bootstrapped critic.
2. **Semantic correctness is judged heavy-vs-heavy at steady state, with the
   switch controlled** (V4-C2R), not A-heavy vs center over a fresh
   rollout. The center comparator mixed in center drift, and the unswitched
   control mixed in the switch transient.
3. **The target is objective specificity for every pair**:
   S_i(w_i⁺) > S_i(w_j⁺) for each j ≠ i, not the mean over j. The mean hid
   A ≈ O in V4-C2R.

## Per run (iteration 300)

A. **Training integrity:** 300 iterations, no stop gate (non-finite,
   log-std, dead critic).

B. **m = 3 anchor, specificity (primary):** the V4-C2R protocol with K = 3.
   512 snapshots, center warm-up, discarded burn-in, then four 128-step
   branches (T⁺, A⁺, O⁺ and a repeat of one) with cyclic position balance.
   Heavy = 0.70 on i and 0.15 on each of the others. Steady window 33–128. For each
   ordered pair (i, j), D_ij = S_i(i⁺) − S_i(j⁺); one-sided 95% simultaneous
   bounds over the six ordered pairs. Pair correct if LCB > δ₃, wrong if
   UCB < −δ₃. **Anchor passes if all six ordered pairs are correct.**
   Fidelity invalid if |repeat-null mean| > δ₃ for any objective.
   δ₃ = max(0.002, 1.2 × max |mean null|) from a new switch-matched null
   (K = 3) on two V4-C3 checkpoints (the first G1-1 and G1-2 seeds), run
   before any specificity result.

C. **m = 3 anchor, critic:** MC256 critic validity per objective (EV mean > 0,
   negative fraction ≤ 0.25, registered pooling).

D. **Held-out cardinality:** the same specificity test on the held-out sets.
   - G1-1 (held out m = 1): the singleton sets {i}. D_ij = S_i({i}) − S_i({j})
     for i ≠ j, six ordered pairs.
   - G1-2 (held out m = 2): each pair set {i, j} with its two heavy endpoints
     (0.70 / 0.30). D = S_i(i-heavy) − S_i(j-heavy) and the reverse, two
     ordered pairs per set, six over the three sets.
   Held-out passes if all six ordered pairs are correct.

E. **Seen non-anchor cardinality:** the same test (G1-1: m = 2; G1-2: m = 1),
   reported.

F. **Continuity (descriptive):** the V4-C m = 3 anchor endpoint protocol (heavy
   vs center, 4 suites × 64 steps, G1 rule) and preference authority vs V3 G0.

## Cross-seed

Per fold, k/3 runs passing A + B + C, and k/3 passing D. The objective layer
is frozen as valid if both folds pass B and C in ≥ 2/3 runs; held-out
generalization is claimed only if D passes in ≥ 2/3 runs in both folds.

## Open before freezing

- Whether B requires all six ordered pairs, or allows the A–O pair to be
  flat (V4-C2F: A/O distinct, specificity only partly realized).
- The thresholds for "the objective layer is valid" (≥ 2/3 per fold as drafted).
