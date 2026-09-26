# L0-A stock reproduction

Status: **STOCK REFERENCE SCREEN PASS — 3 seeds**

The installed Isaac Lab `Isaac-Velocity-Flat-Unitree-A1-v0` task was trained with the installed `rsl_rl` runner and stock `UnitreeA1FlatPPORunnerCfg` for 300 iterations, 4096 environments, seeds 0, 1, and 2. The summary is [L0A_3SEED_SUMMARY.json](/home/xero/Master's%20Degree/Thesis/talon-rl/artifacts/l0a_screen/L0A_3SEED_SUMMARY.json).

| seed | final mean episode length | velocity XY error | base contact | final learned std |
|---:|---:|---:|---:|---:|
| 0 | 980.8 | 0.194 | 0.0145 | 0.329 |
| 1 | 995.4 | 0.205 | 0.0067 | 0.318 |
| 2 | 981.5 | 0.172 | 0.0052 | 0.292 |

All three seeds show the same nominal-task learning pattern: episode length approaches the 1000-step horizon, timeout dominates base-contact termination, velocity error falls, and learned standard deviation anneals from approximately 1.0. This is a reference sanity result only; it is not a Gate-0B verdict because the stock task/reward/observation/command contract is different.

## Interpretation

L0-A makes the low-level Isaac Lab + A1 + physics + stock `rsl_rl` stack substantially less likely to be the sole failure source. Combined with L0-B failing on the unchanged Gate-0B task, evidence weight shifts toward Gate-0B custom task/reward/observation/reset/command semantics and their seed-dependent basin landscape. The next justified experiment is instrumented Gate-0B reproduction (L0-C1), without changing its formulation.
