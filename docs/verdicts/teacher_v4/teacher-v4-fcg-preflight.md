# Teacher V4 — FC-G live-rollout preflight

Status: **PASS, 2026-10-01. No FC-G 50-iteration outcome branch has run.** The [FC-G contract](../../contracts/teacher_v4/teacher-v4-fcg-gradient-composition-contract.md) and arms were frozen at `bb4f974` before this preflight. A follow-up implementation commit only adds the registered per-minibatch audit output. Raw artifacts: `runs/teacher_v4_fc-2026-10-01/fcg_preflight/`.

Each run resumed the registered u0 checkpoint, used 4096 environments, seed + 10⁶ + 1000×11, and stopped after one update. The real rollout had 98,304 rows and each PPO minibatch had 4,096 rows. Unit preflight: 29 PPO/FC-F/FC-G tests passed, including GLOBAL versus legacy after loaded Adam history and SPLIT's unchanged critic step on a shared minibatch.

| Seed | GLOBAL vs legacy after one update | Maximum stream-sum gradient error | `c_T/c_N` mean in GLOBAL | `c_global < c_N` fraction in GLOBAL |
|---|---|---:|---:|---:|
| 79101 | model, actor Adam, critic Adam, LR: exact | 4.37×10⁻⁷ | 0.391 | 1.00 |
| 79103 | model, actor Adam, critic Adam, LR: exact | 6.00×10⁻⁷ | 0.316 | 1.00 |

The two dual vectors differed by at most 1.8×10⁻⁸ across separate simulator launches; the optimizer equivalence is exact. The `GLOBAL` arm uses the legacy backward and clip path; stream decomposition is audit only. The online dual computation is the same in all arms.

| Seed / arm | Per-minibatch audit rows | Range `c_T/c_N` | Range `cos(G0,G2)` | Fraction with >5% coefficient difference | Maximum G2 norm-match error |
|---|---:|---:|---:|---:|---:|
| 79101 / SPLIT-NORM-MATCHED | 20 | 0.253–0.604 | 0.99160–0.99805 | 1.00 | 7.56×10⁻⁸ |
| 79103 / SPLIT | 20 | 0.140–0.550 | 0.97497–0.99133 | 1.00 | n/a |
| 79103 / SPLIT-NORM-MATCHED | 20 | 0.168–0.550 | 0.96929–0.99133 | 1.00 | 7.08×10⁻⁸ |

All three arms completed a real first update at seed 79101 and seed 79103. The cosine ranges show a modest but nonzero direction change; the coefficient ratio and within-minibatch suppression show that the clip treatments were actually exercised. No endpoint claim follows from this preflight. The registered r = 11–13 screen remains the next step, with r = 14–15 only if open.
