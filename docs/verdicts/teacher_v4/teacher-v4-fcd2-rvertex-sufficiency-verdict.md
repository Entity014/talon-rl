# Teacher V4 — FC-D2 R-Vertex Sufficiency Verdict

Status: **FROZEN. Registered verdict: "R-vertex task pressure not sufficient alone in 79101". In 79101, D2 (R-vertex λ only) is not inverted: −24.4 / −9.1 %. Descriptively, D2p (R⁺ λ only) *is* inverted: +15.5 / +15.9 %. The near-repeats disagree: 79102 D2 inverts, 79103 D2 does not. The inversion is therefore branch-sensitive at the ~7-point level.**
Date: 2026-09-30
Contract: [teacher-v4-fcd2-rvertex-sufficiency-contract.md](../../contracts/teacher_v4/teacher-v4-fcd2-rvertex-sufficiency-contract.md) (frozen at `36a8379`, before any D2 branch was trained)
Output: `runs/teacher_v4_fc-2026-09-30/fcd2/` (`seed<s>/D2/`, `seed79101/D2p/`, `fcd2_probe.json`, `fcd2.out`)

**Check passed.** Every k = 0 replay equals A's (difference 0.0).

## Registered result (79101, primary)

Endpoint = mean over k = 30, 40, 50. Values are raw Δ_R / matched Δ_R.

| arm | λ in loss (branch mean) | endpoint | Δtl endpoint | state |
|---|---|---|---|---|
| A (FC-C) | all regions: R⁺ 2.22, R-vertex 14.58, C 0.16 | +14.1 / +20.0 % | −0.04 to −0.06 | inverted |
| **D2** | R vertex only: 14.60 | **−24.4 / −9.1 %** | −0.15 | **not inverted** |
| D2p (secondary) | R⁺ only: 2.07 | +15.5 / +15.9 % | −0.04 to −0.05 | inverted |

The dual trajectories are almost unchanged by the loss mask (R-vertex λ is
14.6 in every arm). So the arms differ in which λ enters the update, not
in the λ values.

## Near-repeats (79102, 79103: consistency check, no vote)

In these seeds, A's λ outside the R vertex was ~0. So D2 and A have
practically the same loss.

| seed | A | D2 (near-repeat) |
|---|---|---|
| 79102 | +7.6 / +10.2 % inverted | +7.8 / +7.5 % inverted |
| 79103 | +10.1 / +9.4 % inverted | **+2.7 / +2.9 % not inverted** |

## Descriptive (not registered)

- **In 79101, the task pressure that reproduces the inversion is at R⁺
  itself.** R⁺-only λ inverts, as strongly as A. R-vertex-only λ does not,
  and looks like D1 (λ ≡ 0): R⁺ tracks worse (Δtl −0.15) and rotates less.
  FC-C arm B (fixed R⁺ with λ_R⁺) did *not* invert. So in 79101 the
  inversion needs R⁺ task pressure together with mixed training.
- **Branch noise is about 7 points of Δ_R.** In 79103, two branches with a
  practically identical loss end at +10 % and +3 %. So a single branch near
  the gate is not decisive. This also bears on earlier single-branch
  results:
  - FC-C fixed-R⁺ arms: −3 to −18 %.
  - 79101 D1 (−26 / −12 %) and D2 (−24 / −9 %) against A (+14 / +20 %). These
    gaps are well beyond 7 points.
  - 79103 D1 (−8 / −9 %) against the A / D2 pair (+10, +3 %). The gap is at
    least 11 points.
  - 79102 D1 (+1.4 / +3.5 %) against the A / D2 pair (+8 / +10, +8 / +8 %).
    The gap is about 4–7 points, which is within noise.
- **Revised picture.** The common factor is not "the R vertex". It is task
  pressure in the R-heavy part of the preference space, applied during
  mixed training:
  - In 79101, it is R⁺'s own λ.
  - In 79102 and 79103, the R vertex is the only candidate. Its effect is
    reproducible in 79102 and marginal in 79103.
  - The R-vertex-to-R⁺ leakage hypothesis from FC-D1 is **not supported**
    in 79101.
- **O control holds** in every D2 / D2p arm (Δ_O −5 to −25 %).

## Reading

- As registered: R-vertex task pressure is not sufficient on its own in
  the one seed that could test it.
- Descriptively, the inversion follows task pressure on R⁺ (79101) and is
  branch-sensitive where the pressure is only at the R vertex (79102 vs
  79103). "An infeasible R vertex drives the failure" is weakened. "Task
  pressure on the R-heavy side, under mixed training, raises R⁺ rotation
  relative to C" remains.
- **Before any further causal claim, the next contract needs repeat
  branches.** Several run seeds per arm are needed to measure branch
  variance, because the current effect size (~10 points) is not far above
  the observed noise (~7 points). FC-D3 (R-vertex feasibility) is less
  central now, but it is still a separate, valid question. Parameter
  attribution is premature until the arm contrast is repeated.

**Limits.** Sufficiency tested in one seed. One branch per arm. 50
iterations. The near-repeats are two samples of branch noise, not a
variance estimate.
