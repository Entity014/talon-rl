# Authority-Isolated Coordinate-Aware Tail Repair Verdict

Status: **FROZEN — FAIL; TOTAL-LOGIT TAIL PENALTY REDUCES TAILS BUT DESTROYS AUTHORITY AND ROBUSTNESS**
Date: 2026-09-25

## Paired screen

Both arms started from the same u75 actor + repaired wide critic and used identical:
- GAE lambda = 0.95
- preference schedule
- Foundation/support refresh
- optimizer / learning rate
- rollout seeds
- 25 updates

CONTROL:
    no tail gradient

TREATMENT:
    per-action q95 tail budgets from robust u50 fixed probes
    squared raw-logit excess penalty
    gradient-budget cap rho = 0.25

## Start match

Paired u0 authority metrics matched exactly:

    pairwise action separation    0.5293
    tangent Jacobian              1.4515

The treatment therefore began from the same actor function.

## Authority trajectory

Relative to each arm's identical u0:

    update       control pair/jac ret      treatment pair/jac ret
      0             1.000 / 1.000             1.000 / 1.000
      5             0.522 / 0.511             0.576 / 0.546
     10             0.497 / 0.483             0.482 / 0.468
     20             0.742 / 0.715             0.368 / 0.372
     25             0.785 / 0.746             0.403 / 0.412

At u25 the treatment also retains only about half of the paired control's remaining authority.

Thus the authority-preservation gate fails decisively.

## Robustness trajectory

Suites 2-3:

    update        control min/fails       treatment min/fails
      0              0.875 / 7               0.875 / 7
      5              0.625 / 14              0.750 / 12
     10              0.625 / 12              0.250 / 40
     20              0.750 / 7               0.125 / 44
     25              0.750 / 9               0.125 / 43

Final treatment robustness is much worse than paired control.

The residual suite3/C lane is not rescued:

    control u25 suite3/C survival      0.75
    treatment u25 suite3/C survival    0.50

FL-hip t4..7 remains saturated at +1.0 in both arms, so preserving that single support coordinate is not sufficient once the rest of the trained action geometry drifts.

## Tail mechanism

Mean fixed-probe excess above per-action tau:

    start       1.547
    control25  11.697
    treatment25 6.201

Active tail fraction:

    start       0.375
    control25   0.692
    treatment25 0.645

The treatment does reduce tail growth relative to control, but it does not restore the initial operating regime.

Per-coordinate redistribution is also nonuniform. Several coordinates reduce strongly, while others increase relative to control, including RL thigh and some calf coordinates.

## Gradient budget

The adaptive tail multiplier ranged around 0.002-0.021, but by construction the effective tail-gradient norm was exactly:

    ||alpha g_tail|| / ||g_PPO|| = 0.25

at every treatment update.

Therefore the failure cannot be interpreted as an accidental uncapped regularizer-gradient explosion.

## Interpretation

The candidate fails all substantive repair gates except "tail excess reduced."

This establishes that penalizing the **total preference-conditioned raw logits** is not a suitable repair direction:
- it directly competes with learned preference-specific output geometry;
- it reduces authority much more than the paired control;
- it does not sufficiently suppress PPO-driven tail growth;
- robustness collapses despite lower tail excess.

The result argues against merely tuning rho as the next move. Lower rho would weaken an already insufficient tail correction, while higher rho would intensify the demonstrated authority conflict.

## Next mechanistic candidate

The next repair should change *where* the tail pressure acts, not merely its scalar strength.

Most causal-clean candidate:

    common-mode / center-path coordinate-aware headroom repair

For each state, decompose output conceptually as:

    z_common(s) = z(s, w_center)
    delta_z(s,w) = z(s,w) - z_common(s)

Apply per-coordinate tail pressure to z_common only, while leaving preference-specific delta_z outside the regularizer gradient path.

This targets shared saturation / loss of headroom without directly shrinking the preference-family differential that defines authority.

The same paired authority, residual-lane, and suite2/3 robustness gates should remain frozen.

AI-C2 / H2 remain blocked.
