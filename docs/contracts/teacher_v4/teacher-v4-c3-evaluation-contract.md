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

B. **m = 3 anchor, semantics (primary).** Revised before any evaluation
   (2026-09-28): requiring S_i(i⁺) > S_i(j⁺) for every ordered pair would
   treat synergy (A⁺ also improving O) as failure. Cross-objective
   improvement is allowed. The two requirements are:

   Protocol: V4-C2R with K = 3 and the center included. 512 snapshots, center
   warm-up, a discarded burn-in, then five 128-step branches: C, T⁺, A⁺, O⁺,
   and a repeat of one heavy preference, with cyclic position balance. Every
   branch is restored in place, and every heavy branch switches at t = 0.
   Heavy = 0.70 on i and 0.15 on each of the others. Steady window 33–128.
   Response vector of preference j: r_j = S(j⁺) − S(C) = [ΔS_T, ΔS_A, ΔS_O].

   - **B1 — self-direction:** for each i, S_i(i⁺) − S_i(C). One-sided 95%
     simultaneous bounds over the three objectives. Correct if LCB > δ₃.
     B1 passes if all three are correct.
   - **B2 — identifiability:** for each unordered pair (i, j), the vector
     r_i − r_j = S(i⁺) − S(j⁺) over the three scores. Two-sided 95%
     simultaneous bounds over the 3 pairs × 3 components. The pair is
     distinguishable if at least one component has LCB > δ₃ or UCB < −δ₃, in
     either sign. B2 passes if all three pairs are distinguishable. A pair
     with every component inside ±δ₃ is a collapse (redundant preferences).
   - **Anchor B passes if B1 and B2 pass.**
   - Reported, not gated: the full interaction matrix M[score, preference]
     (off-diagonal positive = synergy, negative = conflict, ≈ 0 =
     independence), the ordered-pair specificity S_i(i⁺) − S_i(j⁺), and the
     B1 transient-window values. The transient values show whether the
     unswitched center reference leaves a switch artifact.

   Fidelity invalid if |repeat-null mean| > δ₃ for any objective.
   δ₃ = max(0.002, 1.2 × max |mean null|) from a switch-matched null (K = 3)
   on two V4-C3 checkpoints (G1-1 s73101, G1-2 s73101), run before any
   semantics result.

C. **m = 3 anchor, critic:** MC256 critic validity per objective (EV mean > 0,
   negative fraction ≤ 0.25, registered pooling).

D. **Held-out cardinality:** the B1/B2 logic applied to the held-out sets.
   - G1-1 (held out m = 1): singleton sets {i} against the m = 3 center.
     B1: S_i({i}) − S_i(C) > δ₃ for each i. B2: {i} vs {j} distinguishable,
     using the same component rule.
   - G1-2 (held out m = 2): for each pair set {i, j}, its two heavy endpoints
     (0.70 / 0.30) against the m = 3 center. B1 for i and j. B2 between the
     two endpoints.
   Held-out passes if B1 and B2 both pass.

E. **Seen non-anchor cardinality:** the same test (G1-1: m = 2; G1-2: m = 1),
   reported.

F. **Continuity (descriptive):** the V4-C m = 3 anchor endpoint protocol (heavy
   vs center, 4 suites × 64 steps, G1 rule) and preference authority vs V3 G0.

## Cross-seed

Per fold, k/3 runs passing the objective-layer gates and k/3 passing D.
The objective layer is valid if its gates pass in ≥ 2/3 runs in both folds.
Held-out generalization is claimed only if D passes in ≥ 2/3 runs in both
folds.

## Decisions recorded (2026-09-28)

- Anchor B = B1 (self-direction) + B2 (identifiability). Synergy allowed.
- Objective layer valid: passes in ≥ 2/3 runs in both folds. D is a separate
  generalization claim.
- G1-1 s73102 passed the preregistered training-integrity gate but showed a
  prolonged near-collapse of the critic representation (iterations 22–171).
  It is included in all primary analyses. Its exclusion is reported only as
  a post-hoc sensitivity analysis and never changes the verdict.

## Open before freezing

- Which gates make up "objective layer valid": B + C (critic MC256 at the
  anchor), with E (seen non-anchor) reported, or B + C + E.
