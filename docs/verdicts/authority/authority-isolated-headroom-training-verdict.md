# Authority-Isolated Headroom Training Verdict

Status: **FROZEN — PARTIAL REPAIR; AI-C2/H2 NOT YET AUTHORIZED**
Date: 2026-09-25

## Branch question

Can actor output/headroom robustness be repaired without reducing preference authority or sacrificing actuator support magnitude?

## Per-action tail budgets

Per-coordinate raw-logit tail budgets tau_j were calibrated from the robust u50 actor on the fixed probe corpus (q95 of |z_j|), not from the residual FL-hip lane.

This confirmed strongly different natural operating ranges across actuators and justified coordinate-aware rather than global thresholds.

## Fixed lambda treatment: FAIL

Treatment:

    L = L_PPO + 0.01 L_tail

At update 0:
- raw tail gradient norm: 0.833
- effective lambda=.01 gradient norm: 0.0083
- PPO gradient norm at update 1: ~14.6

Thus the tail correction initially had only about 0.057% of PPO gradient magnitude.

At u10:
- pairwise authority retention: 74.6%
- tangent Jacobian retention: 60.7%
- probe tail loss: 38.35x initial
- suites2/3 failed lanes: 32
- minimum survival: 0.375

This treatment is rejected.

## Gradient mechanism audit

The static fixed-probe tail gradient was mildly aligned with an authority-increasing preference-separation gradient:

    cosine full actor:   +0.226
    actor trunk:         +0.196
    family path:         +0.255

Therefore the tail objective is not intrinsically anti-authority.

However, on actual PPO update batches the PPO and tail gradients were strongly conflicting:

    update 1 cosine  -0.757
    update 2 cosine  -0.529
    update 3 cosine  -0.187

A 10% gradient-budgeted tail term preserved authority at u3 but could not control tail growth because it was too small to neutralize the conflicting PPO component.

## Minimal conflict projection

The next treatment removed only the PPO-gradient component that would increase tail loss:

    g_proj =
      g_ppo - min( <g_ppo,g_tail>/||g_tail||^2, 0 ) g_tail

Pure projection preserved/grow authority, but fixed-probe tail still increased 1.41x and semantic robustness did not improve sufficiently.

## Projected + active tail descent

Final screened treatment:

    g_proj = conflict-projected PPO gradient

    g =
      g_proj
      + 0.05 * ||g_proj||/(||g_tail||+eps) * g_tail

Everything else remained frozen:
- authority-isolated actor
- wide critic
- GAE lambda=.95
- Foundation/reset-diverse support
- preference schedule
- tau_j
- optimizer/lr
- stochastic PPO contract

### Training result at u10

All updates 4..10 had training termination fraction 0.

Authority:

    pairwise action separation retention   1.448
    simplex-tangent Jacobian retention     1.634

Tail:

    fixed-probe tail-loss ratio            1.226
    fixed-probe tail-fraction delta       -0.0049

The tail loss magnitude is still somewhat above u0, but the fraction of probe coordinates beyond their per-action q95 budgets is slightly below u0.

The PPO/tail conflict remained large and state-dependent. At update 10:

    cosine(g_ppo,g_tail)                  -0.819
    removed PPO-gradient norm fraction     0.819

This confirms that ordinary PPO strongly pushes toward the extreme-logit regime in some updates.

## Semantic reset robustness

Suites 2 and 3, 8 lanes each preference:

    checkpoint / treatment             failed lanes   min survival
    frozen u75 actor                         9             0.875
    ordinary control u10                    14             0.625
    fixed-lambda tail u10                   32             0.375
    pure projection u3                      10             0.625
    projected+descent u3                     8             0.750
    projected+descent u10                    5             0.750

Projected+descent u10 failures:

    suite2 / O : lane 2, lane 7
    suite3 / O : lane 0
    suite3 / S : lane 0
    suite3 / C : lane 0

All are base_contact.

Recovered relative to frozen u75 include:
- suite2 A
- suite2 S
- suite2 C
- suite3 T
- suite3 A

Suite2/O remains the main broad residual and gains one additional failing lane.

## Residual support margin

suite3/C/lane0 FL-hip action at t=4..7:

    0.99985
    0.99958
    0.99783
    0.99181

Thus the repair does not fail by suppressing the previously identified FL-hip support margin.

## Decision

PASS:
- preference authority preserved
- actuator support margin preserved
- tail fraction controlled
- failure count improves substantially versus ordinary PPO continuation
- failure count improves versus frozen u75 (9 -> 5)

FAIL:
- global robustness gate min survival >= .95
- zero/near-zero residual semantic-suite failures

Therefore:

    projected + tail descent = PARTIAL MECHANISTIC REPAIR
    AI-C2                     = BLOCKED
    AI-H2                     = BLOCKED

## Next question

Do not tune kappa further yet.

The remaining failure topology has changed and is now concentrated:
- orientation-heavy preference O, especially suite2
- suite3 lane0 under O/S/C

The next step should be a read-only residual audit of these remaining regimes before any further training intervention. Determine whether the residual is still raw-logit/headroom conflict, a different actuator support threshold, or an objective-specific state/dynamics regime.
