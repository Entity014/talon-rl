# Teacher V4 — V4-C G1 Cardinality-Generalization Contract

Status: **PREDECLARED — FROZEN 2026-09-27, before any V4-C training run finished and before any evaluation**
Date: 2026-09-27

## Question

Same as V3 G1 ([objective-set-g1-variable-cardinality-contract.md](../objective_set/objective-set-g1-variable-cardinality-contract.md)):
can one objective-set-conditioned controller, trained on some active-set
cardinalities, generalize zero-shot to a held-out cardinality? V4-C asks it
of TeacherV4 trained from scratch, on the substrate V3 used, so the result
is comparable with the V3 G1 verdict (seen-support semantic learning failed
there, so held-out cardinality was never cleanly tested).

## Training (frozen elsewhere, restated)

- Env `Isaac-Talon-A1-V4C-v0` (stock A1 flat + e_t), objective contract
  T/A/O/S with the T3-B divisors, action contract scale 0.25 / Kp 25 / Kd 0.5:
  [teacher-v4-c0a-objective-contract-verdict.md](../../verdicts/teacher_v4/teacher-v4-c0a-objective-contract-verdict.md).
- Sampling/optimization: exact M0, 4096 × 24, 5 epochs × 4 minibatches, 300
  iterations = 29,491,200 samples:
  [teacher-v4-c0-rollout-batch-audit.md](../../verdicts/teacher_v4/teacher-v4-c0-rollout-batch-audit.md).
- Trainer `train_v4c.py` at commit `822cbcb`; loss has no edge retention,
  tail projection, reference anchoring or semantic loss.
- Folds and seeds as V3 G1: G1-2 trains {3,4}, holds out 2; G1-3 trains
  {2,4}, holds out 3; seeds 73101, 73102, 73103. All six runs required; no
  seed replaced.

## Declared differences from V3 G1

| | V3 G1 | V4-C G1 | why |
|---|---|---|---|
| init | G0 (Phase-1 weights) | random (from scratch) | TeacherV4 shapes differ |
| budget | 8 × 32 × 30 updates | 4096 × 24 × 300 iterations | from-scratch regime = M0 |
| set sampling | one cardinality and subset per update, all lanes | per env, redrawn when its episode ends | persistent rollouts |
| edge retention | yes, vs G0 | none | no reference to retain |
| authority gate reference | same fold's G0 init | V3 G0 model (Phase-1 authority level), same probes and sets | V4 has no G0; random-init authority (≈7e-4) is no functional standard; the seed-matched V3 G1 u30 belongs to a branch whose seen-support semantics failed |
| probe e_t | none | one fixed physical nominal stock plant for every state and model (see below) | TeacherV4 needs e_t; plant variation must not enter the authority metric |
| training logs | per-update ids/weights/queried tokens | per-iteration aggregates | held-out exclusion is structural: the sampler draws only from `--cardinalities` (gate `test_objective_set_sampler_contract`) |

## Leakage

The held-out cardinality appears nowhere in training: not in rollouts, critic
targets, sampling, checkpoint choice or any training-time evaluation.
`train_v4c.py` evaluates nothing semantic during training.

## Checkpoint

Iteration 300 (`model_300.pt`), fixed in advance. Iterations 50…250 are
diagnostics only. No semantic, held-out or engineering result selects a
checkpoint.

## Evaluation

`scripts/rl/experiments/architectures/authority/teacher_v4/g1_evaluate.py`,
a port of `objective_set_g1_evaluate.py` with only the model API, the env id,
the authority reference and the probe plant context changed. Everything else is identical: 8 envs ×
64 steps × 4 suites, suite seeds, all m = 2, 3, 4 sets, endpoint / center /
continuum / interior / critic / survival criteria and thresholds, fixed
probe states, permutation tolerance 1e-6, authority threshold 0.75.

Per fold/seed:

1. **Training integrity:** run completed 300 iterations with no stop gate.
2. **Full-set anchor (m = 4):** T, A, O endpoint semantics PASS, critic valid,
   endpoint survival ≥ 0.95. S reported only.
3. **Seen cardinalities:** all sets evaluated and reported.
4. **Held-out cardinality:** every held-out set passes all criteria
   (fold-seed pass requires anchor pass and held-out pass).
5. **Cross-seed:** reported as k/3 per fold. G1 is supported only if both
   folds pass on the three-seed characterization, as in V3 G1.

## Authority gate

Primary (gated), per evaluated set and per metric, exactly as V3 G1:

    pairwise authority(V4, iteration 300) >= 0.75 × pairwise authority(V3 G0)
    tangent Jacobian authority(V4, iteration 300) >= 0.75 × tangent authority(V3 G0)

Reference `runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt`,
probe states `runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz`,
the same active sets and preferences for both models.

Plant context for V4 on the probes: the **physical nominal stock plant**,
not normalized zeros. Normalized zeros would be the training mean of e_t,
which includes the +1 kg mean of stock `add_base_mass` (−1…+3 kg). The
evaluator takes the live e_t (in V4-C every channel but trunk mass is
constant, which it asserts), sets trunk mass to the asset's
pre-randomization mass (`robot.data.default_mass`), and passes it through
the checkpoint's frozen e_t normalizer. The raw and normalized vectors are
written to the report. V3 G0 has no e_t and sees the same probe states.

Descriptive only, never gated: V4's authority at initialization (rebuilt
from the training seed exactly as `train_v4c.py` builds it) and the growth
ratio iteration 300 / init.
