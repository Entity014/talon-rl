# Teacher V4 — V4-C G1 Verdict

Status: **FROZEN — G1 FAIL (0/6 fold-seeds) under the preregistered contract; critic-validity gate failed on 66/66 sets; seen-support semantics substantially stronger than V3 G1 (descriptive)**
Date: 2026-09-27
Contract: [teacher-v4-c-g1-contract.md](../../contracts/teacher_v4/teacher-v4-c-g1-contract.md) (frozen before evaluation, commit `77e34f3`)

## Runs

Six runs, `runs/teacher_v4_c_g1_{2,3}_seed{73101,73102,73103}-2026-09-27/`,
`train_v4c.py` at `822cbcb`, 300 iterations each (29,491,200 samples),
evaluated at `model_300.pt` with `g1_evaluate.py` (`g1_evaluation.json` per
run). No seed replaced, no checkpoint chosen, no threshold changed.

Provenance, `normalizer_impl`:
- training: `RunningMeanStd` batch statistics in float32 (pre-`61a483d`);
  constant e_t channels normalized to a constant offset (friction 0.536,
  dynamic friction −0.983) instead of 0, identical for every sample in a run;
- evaluation: the statistics saved in each checkpoint;
- canonical from `61a483d`: float64 batch statistics, constant channels → 0.

## Preregistered gates

| gate | result |
|---|---|
| A. training integrity | **6/6 pass**: 300 iterations, no stop gate, KL ≈ 0.01, training-time EV 0.97–1.00 on all four objectives |
| B. m=4 anchor (T/A/O semantics + critic valid + survival ≥ 0.95) | **0/6** — critic invalid in all six; T/A/O endpoint semantics pass in 2/6 (G1-2 s73102, G1-3 s73101); A is the failing endpoint in the other four |
| C. seen cardinalities | 0 of 30 seen sets pass (critic invalid on all) |
| D. held-out cardinality | 0 of 30 held-out sets pass (critic invalid on all) |
| E. cross-seed | G1-2 0/3, G1-3 0/3 → **G1 not supported** |

Criterion pass rates over all 66 evaluated sets:

| criterion | pass |
|---|---:|
| required T/A/O semantics | 36/66 |
| center compromise | 56/66 |
| continuum monotonicity | 65/66 |
| continuum between | 60/66 |
| **critic valid (EV > 0, negative fraction ≤ 0.25)** | **0/66** |
| endpoint survival ≥ 0.95 | 59/66 |
| authority pairwise ≥ 0.75 × V3 G0 | 56/66 |
| authority tangent ≥ 0.75 × V3 G0 | 56/66 |
| permutation ≤ 1e-6 | 65/66 |

Authority mostly exceeds the Phase-1 reference (retention often 1.0–3.4);
failures concentrate on the {T,O} and {A,O} sets. The single permutation
failure is G1-3 s73101 {T,A,S}.

## Descriptive only (not the verdict)

Setting the critic gate aside, sets passing every other criterion:

| run | seen | held-out | m=4 anchor without critic |
|---|---:|---:|---|
| G1-2 s73101 | 3/5 | 1/6 | fail (A) |
| G1-2 s73102 | 5/5 | 2/6 | pass |
| G1-2 s73103 | 3/5 | 3/6 | fail (A) |
| G1-3 s73101 | 4/7 | 1/4 | pass |
| G1-3 s73102 | 2/7 | 1/4 | fail (A) |
| G1-3 s73103 | 1/7 | 1/4 | fail (A, S survival) |

Against V3 G1, whose verdict reports systematic seen-support T/A/O failure
(e.g. seed 73101 m=4: T, A, O all fail), V4-C's seen-support semantics are
much stronger: T and O endpoints pass almost everywhere. The remaining
semantic failure is concentrated in **A (angular stability)**, and in sets
pairing A or S where survival sometimes drops (G1-3 s73103).

## Critic-validity gate: likely a metric/regime mismatch, not yet tested

The gate compares the query critic's V with **truncated 32-step discounted
returns without bootstrap** (`segret` over steps 0–31 and 32–63). V4's
critic was trained on persistent 1000-step episodes with GAE targets that
bootstrap the tail, so V estimates the long-horizon return. The truncated
target shrinks toward the segment end while V does not, which alone
produces large error variance relative to the small across-env variance of
the returns of a stable policy. Consistent with this: training-time EV on
GAE targets is 0.97–1.00 for all four objectives, the mean absolute bias at
evaluation is smaller than V3's (median 0.30 vs 1.78), and the V3 critic,
trained on 32-step rollouts from reset, matched this target by construction.

This is a hypothesis. It does not change the verdict. A diagnostic that
would test it without touching the frozen gate: recompute EV on the same
rollouts with the bootstrapped target `segret + γ^(32−t) V(s_32)`.

## What this establishes

1. TeacherV4 trains from scratch on the M0 substrate stably in 6/6 runs.
2. Under the frozen V3 G1 protocol, V4-C G1 fails, as V3 G1 did.
3. Unlike V3, the failure is not broad seen-support semantic collapse: it is
   (a) the critic gate everywhere, whose validity for this training regime is
   in question, and (b) the A objective in 4/6 anchors.
4. Held-out cardinality generalization is still not established.

## Post-hoc C1-D0 — critic-target diagnostic (2026-09-27, does not change the verdict)

Script `scripts/rl/experiments/architectures/authority/teacher_v4/critic_target_diagnostic.py`
(commit `654e9af`), output `critic_target_diagnostic.json` in each run
directory. It re-runs exactly the rollouts the critic gate uses (center and
heavy endpoints × 4 suites per set, same seeds, deterministic policy); its
truncated-target EV reproduces the stored `g1_evaluation.json` value on all
66 sets. The bootstrapped target adds γ^(end−t)·V_q(s_end) for the same
objective query under the same set, masked if the episode ended inside the
segment.

Gate-style statistic (EV mean > 0 and negative fraction ≤ 0.25):

| target | sets passing |
|---|---:|
| registered, truncated 32-step | 0/66 |
| bootstrapped segment | **56/66** (EV mean median 0.55, range −0.88…0.88) |

Per objective, medians over 42 set × run entries (pooled EV over all samples):

| objective | EV trunc → boot | bias trunc → boot | corr trunc → boot | min boot EV |
|---|---|---|---|---:|
| T | −9.97 → 0.985 | 0.703 → −0.007 | 0.65 → 0.992 | 0.91 |
| A | −1.94 → 0.738 | −0.120 → −0.006 | 0.35 → 0.880 | 0.03 |
| O | −0.24 → 0.685 | −0.088 → 0.009 | 0.51 → 0.838 | 0.26 |
| S | −41.65 → 0.856 | −0.119 → −0.016 | 0.48 → 0.940 | −0.29 |

The ten sets that still fail with the bootstrapped target all contain S or
are {A,O}: {O,S} ×4, {A,S} ×2, {T,A,S} ×2, {A,O,S}, {A,O}. Four are seen
sets, six held-out.

Reading (pattern "A, with a conditional tail" in the terms set before the
diagnostic): the registered truncated target is incompatible with V4's
long-horizon bootstrapped critic, and that mismatch explains the 0/66. With a
target of matching horizon the critic is valid on most sets and near-exact
for T; the residual weakness sits in S-containing sets, so critic validity
is not a blanket property.

Caveat on the bootstrapped target: it contains the critic's own V(s_end),
so near the segment end it partly scores the critic against itself and
flatters EV. Independent confirmation needs a target whose tail weight is
negligible, e.g. a long Monte Carlo return (γ^256 ≈ 0.08 at γ = 0.99). That
belongs in the revised, preregistered critic contract, not in this
diagnostic.

The registered result stands: **V4-C failed the preregistered G1 contract.**
What changes is the mechanism: the 0/66 critic failure is driven by an
evaluation target incompatible with the long-horizon bootstrapped critic,
not by broad critic collapse.

**Superseded reading (2026-09-27):** the G1-R follow-up with a critic-independent
256-step Monte Carlo target found the critic valid on only 4/66 sets (T valid,
A and S invalid). The 56/66 above was inflated by the bootstrapped target's
self-reference. See [teacher-v4-c-g1r-critic-verdict.md](teacher-v4-c-g1r-critic-verdict.md).
