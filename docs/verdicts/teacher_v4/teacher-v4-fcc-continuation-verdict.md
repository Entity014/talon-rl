# Teacher V4 — FC-C Short Counterfactual Continuation Verdict

Status: **FROZEN. Registered verdict: source named, "cross-preference / shared-parameter interference" (2/3 seeds, 79101 and 79103). 79102 is unresolved at arm D (split: raw −5.6 %, matched +5.7 %). Arm A (the plain mixed continuation) reproduces the inversion in 3/3 seeds, and the inversion survives tracking matching. No fixed-R⁺ arm (B, C or D) inverts in any seed.**
Date: 2026-09-29
Contract: [teacher-v4-fcc-continuation-contract.md](../../contracts/teacher_v4/teacher-v4-fcc-continuation-contract.md) (frozen with `fcc_probe.py` and the `train_v4c.py` flags at `705691b`, before any branch was trained)
Output: `runs/teacher_v4_fc-2026-09-29/fcc/` (`seed<s>/<arm>/`, `fcc_probe.json`, `fcc.out`)

**Determinism check passed.** In every arm and seed, the k = 0 replay equals
FC-B's replay of u0 exactly (difference 0.0).

## Registered result

Endpoint = mean over k = 30, 40, 50. Values are raw Δ_R / matched Δ_R.

| seed (u0) | D: R only | C: R + O | B: + task | A: mixed + dual | reading |
|---|---|---|---|---|---|
| 79101 (450) | −15.7 / −20.3 % not inv. | −17.7 / −16.3 % not inv. | −14.0 / −13.5 % not inv. | **+14.1 / +20.0 % inverted** | A |
| 79102 (300) | −5.6 / +5.7 % **split** | −2.9 / −1.9 % not inv. | −3.0 / −1.5 % not inv. | **+7.6 / +10.2 % inverted** | unresolved at D |
| 79103 (350) | −6.5 / −7.0 % not inv. | −9.2 / −10.1 % not inv. | −9.2 / −10.1 % not inv. | **+10.1 / +9.4 % inverted** | A |

Every endpoint checkpoint is viable (tl ≥ 0.59 at R⁺ and C in every arm).

The 79102 split comes from the R-only arm alone. Both of its numbers sit at
the gate (−5.6 % and +5.7 %). Every other fixed-R⁺ arm of that seed is
clearly not inverted, and its arm A is inverted. The tree stops there as
registered, so this seed stays unresolved.

## Descriptive (not registered)

- **The inversion is not a tracking artefact.** In arm A, matched Δ_R is
  +9 to +20 %, with Δtl between −0.06 and +0.01. At equal tracking, R⁺
  rotates more than C. This answers the tracking confound FC-B left open, at
  least for these continuations.
- **Timing within arm A.** Raw Δ_R crosses +5 % between k = 20 and k = 30
  in 79101 and 79103, and by k = 10 in 79102. This agrees with FC-B's onsets
  (u0 + 50).
- **Training only at R⁺ also lowers rotation at C, and more.** S_shared
  (F_rate at C) falls 61–69 % in the fixed-R⁺ arms, against 25–42 % in arm
  A. So the shared rotation reduction spreads from R⁺ to the untrained C
  condition. R⁺ still stays at or below C.
- **79103: arms B and C are identical.** λ_R⁺ = 0 at u0 and stays 0, so the
  two losses are the same. With common random numbers the branches match
  bit for bit. In 79102, λ_R⁺ = 0.36 goes to 0 at once, so B ≈ C. Only
  79101 (λ_R⁺ 3.68 at u0) really tests the task arm, and B is not inverted
  there either.
- **O positive control.** It holds in A (Δ_O −8 to −23 %) and in D (−17 to
  −39 %). In B and C it drops to about zero (−6 to +11 %). When training is
  fixed at R⁺ with O in the loss, the O⁺ condition goes untrained. So the O⁺
  vs C gap is not maintained.
- **Without the task stream, tracking erodes in 79101** (C and D: tl at C
  falls to 0.67–0.83). The other seeds keep tl ≥ 0.82.

## Reading

- As registered, the inversion needs training across the mixed preference
  distribution. The R stream alone, R with O, and R with O and the task
  (all at fixed R⁺) do not produce it within 50 iterations. The plain
  continuation does, in all seeds.
- **What "cross-preference" covers here.** Arm A differs from arm B in more
  than one way, so FC-C cannot separate the following:
  1. updates at other preferences (C, O⁺, O vertex, R vertex) moving shared
     parameters in a way that raises R⁺ rotation relative to C;
  2. R⁺'s share of the data (100 % in B, about one region's share in A),
     a dilution effect;
  3. **the task pressure in other regions.** In arm A, λ at the R vertex is
     still 8–14 at u0. So A carries T-dominated updates at the R vertex that
     B does not. Here "task interaction" is excluded only at R⁺ itself.
- This is consistent with FC-B's "shared cost, not preference axis"
  picture. The w-conditioned difference between R⁺ and C is maintained when
  training is fixed at R⁺, and eroded by the mixed distribution.
- A follow-up that separates items 1–3 would be a new contract. Examples:
  mixed training without the R-vertex region, mixed training with λ frozen
  at 0, or R⁺'s share raised within the mixed sampler. FC-C does not decide
  it.

**Limits.** One branch per arm per seed. 50 iterations. A controlled restart
(all envs reset, run seed = seed + 10⁶). In 2/3 seeds arm B is effectively
arm C. 79102 falls on the gate.
