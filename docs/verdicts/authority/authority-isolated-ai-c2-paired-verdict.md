# AI-C2 Paired Actor-Updating Critic Compatibility Verdict

Status: **FROZEN — AI-C2 FAILS AUTHORITY DURABILITY; CRITIC CAPACITY IS NOT THE CAUSAL BLOCKER; AI-H2 REMAINS BLOCKED**
Date: 2026-09-25

## Question

Does the wide critic capacity repair remain compatible with continued learning of the robust high-authority control-u20 actor, relative to the original critic representation, while preserving authority, critic validity, PPO correctness, and robustness?

## Starting point

Authoritative robust checkpoint:

    runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt

At this checkpoint:
- frozen semantic survival = 1.00
- held-out reset survival = 1.00
- wide-critic H32 / MC64 gate = PASS
- preference authority is retained/grown relative to repaired-u10

## Paired design

Both arms:
- exact same control-u20 actor/log_std
- exact same actor optimizer state
- same GAE lambda=.95
- same preference schedule
- same projected PPO/tail repair
- same kappa=.05
- same original reset/training distribution
- same support/freshness schedule
- same actor update budget
- same seed schedule
- critic body frozen during coupled actor learning
- critic head refreshed identically from representative current-policy support

Before actor learning, both critic architectures were re-equilibrated on the same frozen control-u20 policy dataset:
- 6 reset seeds × T/A/O/S/C × 64 steps
- 15,360 samples
- phase-local H32 targets
- 2,500 critic-only steps
- actor max error after prefit = 0 exactly

CONTROL critic:

    52 -> 128 -> 128 -> 128 -> 4

TREATMENT critic:

    52 -> 256 -> 256 -> 128 -> 4

Authoritative paired namespace:

    runs/authority_isolated_ai_c2_clean2_narrow-2026-09-25
    runs/authority_isolated_ai_c2_clean2_wide-2026-09-25

## Coupled training result

### Narrow control

    pairwise authority retention       0.77234
    tangent Jacobian retention         0.61308
    authority gate                     FAIL

    max PPO ratio error                0
    max train termination fraction     0

### Wide treatment

    pairwise authority retention       0.78836
    tangent Jacobian retention         0.66274
    authority gate                     FAIL

    max PPO ratio error                0
    max train termination fraction     0

The wide critic is slightly better numerically on authority retention, but remains far below the predeclared >=.90 gate.

Therefore there is no clean capacity-specific PASS.

## Endpoint value / robustness audit

### Narrow

Frozen semantic suites:

    H32 EV mean               0.44095
    H32 negative fraction     0.000
    MC64 EV mean              0.34260
    MC64 negative fraction    0.000
    min survival              1.00

Held-out reset suites:

    H32 EV mean               0.43934
    MC64 EV mean              0.34756
    min survival              1.00

Fresh critic suites:

    H32 EV mean               0.44308
    H32 negative fraction     0.025
    MC64 EV mean              0.34520
    MC64 negative fraction    0.0375
    min survival              0.875
    failed lanes              1

Fresh failure:
    preference A, seed 9701000, base_contact

### Wide

Frozen semantic suites:

    H32 EV mean               0.46187
    H32 negative fraction     0.000
    MC64 EV mean              0.35357
    MC64 negative fraction    0.000
    min survival              1.00

Held-out reset suites:

    H32 EV mean               0.45014
    MC64 EV mean              0.35609
    min survival              1.00

Fresh critic suites:

    H32 EV mean               0.45263
    H32 negative fraction     0.0125
    MC64 EV mean              0.34597
    MC64 negative fraction    0.0375
    min survival              0.875
    failed lanes              1

Fresh failure:
    preference T, seed 9700000, base_contact

## Continuity control

To test whether the paired failure was caused by critic replacement/re-equilibration, an additional read-only causal control continued the exact original robust control-u20 state:

    exact control-u20 actor
    exact existing wide critic
    exact actor optimizer state
    no critic replacement
    no critic-body re-equilibration
    same actor update rule
    u20 -> u30

Result:

    pairwise authority retention from u20      0.71465
    tangent Jacobian retention from u20        0.76466
    authority gate                             FAIL

    max train termination fraction             0
    PPO ratio invariant                        exact

Thus authority contraction is not caused by critic replacement.

## Causal conclusion

The AI-C2 failure is common-mode across:
- original/narrow critic;
- re-equilibrated wide critic;
- exact wide-critic continuity from the robust u20 checkpoint.

Therefore:

    critic representation capacity as the current blocker     REJECTED
    critic replacement discontinuity as sole blocker          REJECTED
    PPO ratio failure                                         REJECTED
    immediate training survival collapse                      REJECTED

The remaining blocker is:

    authority durability under continued actor learning

The robust u20 checkpoint is a valid high-authority / high-robustness point, but that authority is not stable under another 10 actor updates with the current learning contract.

## Important interpretation

Do not reinterpret this as a new robustness-mechanism problem.

The failure is not:
- residual basin robustness;
- insufficient coverage;
- critic value representation collapse;
- action-neighborhood fragility.

It is a training-dynamics durability issue:

    robust high-authority policy exists
    but continued PPO learning drifts away from that authority regime

This also means u20 should currently be treated as a selected checkpoint, not as proof of asymptotically stable authority ownership.

## Decision

AI-C2:

    FAIL

Reason:

    wide treatment does not satisfy authority-retention gate
    and fresh minimum survival also falls below .95

AI-H2:

    REMAINS BLOCKED under the frozen roadmap

Do not open new robustness mining.

The next branch should stay actor-side and answer only:

    why does authority contract after the robust u20 point under continued learning?

Candidate diagnostic scope:
- authority trajectory versus update number;
- whether preference-family gradients decay / align across objectives;
- whether common PPO gradient increasingly dominates preference-specific family gradients;
- whether a fixed bounded rehearsal/authority-retention constraint already developed earlier prevents the post-u20 drift without changing objective semantics.

The goal is not to invent a new robustness mechanism.
The goal is to make the already-valid u20 authority regime durable enough for AI-C2 -> AI-H2.
