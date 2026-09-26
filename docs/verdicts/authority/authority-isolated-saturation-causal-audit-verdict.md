# Authority-Isolated Saturation Causal Audit Verdict

Status: **FROZEN — GLOBAL PRE-TANH SCALING SHOWS PARTIAL DOSE RESPONSE, PRIMARY GATE FAILS**
Date: 2026-09-25

## Frozen intervention

The u75 actor was not trained or modified. The only intervention was:

    z' = alpha z
    a  = tanh(z')

for alpha = 1.00, 0.90, 0.80, 0.70 on the four frozen semantic reset suites.

## Robustness

The control reproduces the prior result exactly.

    alpha   min survival   failing trajectories
    1.00       0.875              9
    0.90       0.875              6
    0.80       0.875              5
    0.70       0.875              3

Thus global compression produces a monotone reduction in failure count, but no tested alpha restores the predeclared minimum-survival gate >= 0.95.

The remaining alpha=0.70 failures are:
- suite 2, O: base_contact at step 15
- suite 3, O: base_contact at step 14
- suite 3, S: base_contact at step 14

## Saturation / authority

Across all semantic trajectories:

    alpha   sat fraction   scaled pre-tanh
    1.00      0.9191          56.41
    0.90      0.9079          51.01
    0.80      0.8958          45.42
    0.70      0.8813          39.76

Even alpha=0.70 remains strongly saturated; the intervention does not create large action headroom.

Fixed-probe authority retention:

    alpha   pairwise retention   Jacobian retention
    0.90         0.976               0.971
    0.80         0.948               0.937
    0.70         0.913               0.896

alpha=0.80 preserves both frozen authority metrics above 90%, but survival still fails.
alpha=0.70 is just below the 90% Jacobian-retention threshold and also still fails survival.

Fixed-probe action deviation remains modest through alpha=0.80:

    alpha   mean L2 dev   p95 L2 dev
    0.90      0.0393       0.0797
    0.80      0.0863       0.1719
    0.70      0.1433       0.2812

## Interpretation

The frozen primary causal gate is not passed: no alpha achieves min survival >= 0.95 while retaining both authority metrics >= 90%.

However, the intervention is not null. Failure count decreases monotonically from 9 to 3 as pre-tanh scale is reduced. This supports output operating regime / saturation as a causal contributor, but not as a sufficient explanation under simple global scaling.

The result also does not justify rejecting saturation as causal, because even the strongest tested alpha leaves 88% of action coordinates above |a|>=0.95 on average. The ladder did not actually move the actor far out of the saturated regime.

## Decision

- Do not train yet.
- Do not add a generic action penalty.
- Do not reduce preference authority.
- Do not declare saturation alone sufficient.
- Do not open AI-C2/H2.

The next clean causal test is an extreme-selective smooth pre-tanh compression with unit slope near zero, so that large logits are compressed substantially while mid-range policy structure and preference authority are disturbed less than by global scaling.
