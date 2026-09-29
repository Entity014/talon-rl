# Teacher V4 — FA Formulation Audit

Status: **DESCRIPTIVE, 2026-09-29. Normalization decides most of the locomotion region. Under the current T3-B scaling, locomotion beats the best non-locomoting behavior on 39–72 % of the simplex, and needs w_T ≈ 0.41–0.55 at the equal-split line. With stock-relative scaling (the same behaviors), it wins on 89–96 %, with w_T ≈ 0.06–0.08. Inside the locomoting set, R, V and O genuinely trade off (most locomoting behaviors are non-dominated; ρ(R, V) ≈ −0.5).**
Contract: [teacher-v4-fa-formulation-audit-contract.md](../../contracts/teacher_v4/teacher-v4-fa-formulation-audit-contract.md) (frozen at `ff75740`, before the added traces)
Output: `runs/teacher_v4_fa-2026-09-29/fa_geometry.json` (`fa_geometry.py`)

Bank: 47 behaviors, 10 locomoting (tl ≥ 0.40): s73102 C and T⁺, M0, F4
s74102 / s74103 T⁺, and five F8 T⁺. The F8 T⁺ behaviors translate in this
snapshot protocol (tl 0.47–0.54) but not in the reset protocol (0.35–0.44).
The protocol difference is noted.

## Break-even and normalization sensitivity

Equal split of the non-T weights. w_T* is the smallest w_T at which the
behavior beats the best non-locomoting behavior (median over locomoting
behaviors). The share is the fraction of the simplex (0.02 grid) where the
best locomoting behavior wins.

| scaling | K3 (T,R,O) w_T* / share | K4-V1 w_T* / share | K4-V3 w_T* / share |
|---|---|---|---|
| **T3-B** (current) | 0.55 / 0.72 | 0.46 / **0.39** | 0.41 / 0.53 |
| stock-relative | 0.08 / 0.96 | 0.06 / 0.89 | 0.06 / 0.94 |
| bank-range | 0.61 / 0.66 | 0.52 / 0.36 | 0.54 / 0.35 |

Every locomoting behavior wins somewhere (never-wins = 0 in all cells).
Stock-relative V3 uses an arbitrary weight (−1).

## Locomotion-conditioned geometry

- Non-dominated on (R, V1, O): 7 of 10 locomoting behaviors. On
  (R, V3, O): 7 of 10. On (T, R, V, O): 9–10 of 10.
- Spearman across locomoting behaviors: R–V1 −0.53, R–V3 −0.50, R–O +0.41,
  T–R −0.56. The other pairs are near zero, or −0.39 (V3–O).

## Reading

1. **T3-B normalization is the main lever on the standing region.** It
   scales R and O costs up by about 6× and 64× relative to their stock
   weights (and V similarly). Motion then has to beat several amplified
   penalties at once. With the same behaviors under stock-relative scaling,
   locomotion wins almost everywhere. Adding V (K4-V1) shrinks the T3-B
   region further (0.72 → 0.39).
2. **But stock scaling is not a fix.** At stock scale, R / V / O are tiny
   next to T. Locomotion wins because the other objectives barely count,
   which would also remove their preference authority. The two scalings
   trade task viability against R / V / O authority.
3. **Inside locomotion, R / V / O are real trade-off axes.** Most locomoting
   behaviors are non-dominated, R and V are anti-correlated, and faster
   tracking costs rotation.

Together these support the proposed abstraction: **task feasibility
separated from preference objectives.** Tracking as a requirement (or a
fixed-weight task term), with R / V / O normalized and negotiated among
themselves inside the locomoting manifold. The normalization tension
(viability vs authority) is what the current all-objectives-equal
scalarization cannot resolve. A feasibility-first formulation removes it.
This is descriptive support, not a test: the new formulation needs its own
contract and training.

**Limits.** Observed behaviors only, 10 locomoting. Equal-split break-even
is one line through the simplex. Bank-range scaling is data-dependent.
