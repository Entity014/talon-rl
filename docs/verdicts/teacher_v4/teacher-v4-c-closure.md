# Teacher V4 — V4-C Closure

Status: **FROZEN 2026-09-27 — V4-C closed as a partial / negative result. No later work re-opens or re-scores it.**
Git tag: `v4-c-frozen`

## Scope

V4-C: TeacherV4 (state trunk, env-factor encoder, DeepSets objective set,
family residual actor, objective-query critic) trained from scratch on the
M0 substrate (`Isaac-Talon-A1-V4C-v0`, exact M0 sampling and PPO shell),
V3 T/A/O/S objectives, V3 G1 folds.

## Result

    Infrastructure / trainability          PASS
      train from scratch                   16/16 runs, no stop gate
      preference authority                 1.2–2.0 × Phase-1 (V3 G0) in every run
      optimization collapse                none

    Semantic reproducibility (m = 4, 10 new seeds, iteration 300)
      T   PASS      10/10
      A   FAIL       2/10   (A-heavy worse than center in 7/10)
      O   PASS       9/10
      S   PASS       9/10

    Critic validity (MC256, 10 new seeds)
      T   PASS      10/10
      A   FAIL       0/10
      O   PARTIAL    3/10
      S   FAIL       0/10

    Generalized objective-set G1 (preregistered)   FAIL 0/6
    G1-R long-horizon critic validation            FAIL 0/6 (critic 4/66 sets)

    A diagnosis: systematic formulation / optimization limitation, not
    explained by sign, proxy mismatch, weak gradient magnitude, A–O
    redundancy alone, or seed-specific critic credit quality.

## Documents

1. [teacher-v4-a-b-input-contract-verdict.md](teacher-v4-a-b-input-contract-verdict.md): V4-A/B, e_t contract
2. [teacher-v4-c0-rollout-batch-audit.md](teacher-v4-c0-rollout-batch-audit.md): sampling contract
3. [teacher-v4-c0a-objective-contract-verdict.md](teacher-v4-c0a-objective-contract-verdict.md): objective contract, V4-C env
4. [teacher-v4-c1-screen-verdict.md](teacher-v4-c1-screen-verdict.md): trainability screen
5. [teacher-v4-c-g1-verdict.md](teacher-v4-c-g1-verdict.md): G1 (preregistered)
6. [teacher-v4-c-g1r-critic-verdict.md](teacher-v4-c-g1r-critic-verdict.md): G1-R
7. [teacher-v4-c-a-endpoint-decomposition.md](teacher-v4-c-a-endpoint-decomposition.md): A diagnostics (closed)
8. [teacher-v4-c-seed-sensitivity-verdict.md](teacher-v4-c-seed-sensitivity-verdict.md): 10-seed characterization

## Thesis reading

    V3 failure → architecture redesign → V4 authority and T/O/S improve
               → A remains systematically non-robust

The architecture redesign fixed part of the problem. Generalized preference
authority is not the same as generalized semantic correctness.

## What comes next (separate branches, never mixed into V4-C)

- V4-C2 / A-Repair: objective-formulation repair for A, starting with a
  same-state counterfactual audit. Changes only the objective / learning
  formulation.
- V4-D: plant generalization (broader e_t, Talon env, 2048 × 24 contract),
  only after the Teacher's semantic core passes.
