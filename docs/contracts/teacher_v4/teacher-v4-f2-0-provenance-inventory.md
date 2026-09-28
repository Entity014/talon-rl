# Teacher V4 — F2-0 Provenance Inventory (rare gentle gait, V4-C G1-2 s73102)

Status: **INVENTORY ONLY, 2026-09-28.** No metric has been computed for F2. This page fixes which artifacts F2 may use, and what kind of claim each supports.
Branch: `v4-c2-semantic-preservation`
Follows: [F1 verdict](../../verdicts/teacher_v4/teacher-v4-f1-locomotion-viability-verdict.md)

F2 question: *what observable differences, if any, preceded or accompanied
the acquisition of the low-cost locomotion capability in V4-C G1-2 s73102?*

## Target

`runs/teacher_v4_c_g1_2_seed73102-2026-09-27/`. It is the V4-C behavior
F1 found (partial locomotion at C, tl 0.57, R = [0.72, −0.04, −0.03]; T⁺ tl
0.63).

| field | value |
|---|---|
| training script | `scripts/rl/experiments/architectures/authority/teacher_v4/train_v4c.py` |
| code | `822cbcb` (trainer, 2026-09-27 00:12), env `1f380f0`, model `4c9f401`. The run records no commit, so this is inferred from launch time (model_300 at 00:31) and git history. No uncommitted-state record exists. |
| normalizer | **pre**-`61a483d` (float32 RunningMeanStd batch statistics; the fix landed at 01:27) |
| task | Isaac-Talon-A1-V4C-v0, 4096 envs × 24 steps, 300 iterations |
| fold / cardinalities | G1-2, trained {3, 4}, K = 4 (T, A, O, S) |
| training contract | V4-C G1 training (M0-matched PPO shell) |
| checkpoints | model_50 … model_300 (every 50 iterations) |

## Artifact classes

| source_type | artifact | historical claim allowed |
|---|---|---|
| `online_log` | `metrics.jsonl`, one record per iteration (300): lr, surrogate, value loss, entropy, kl, clip_frac, log_std, reward_per_step [T, A, O, S], explained_variance (per objective), preference_authority, plant_authority, termination_fraction, episodes_finished | **yes** |
| `checkpoint` | `model_{50..300}.pt` | no, only as input to replay |
| `checkpoint_replay` | any metric recomputed from a checkpoint on fixed probes or snapshots (e.g. gait, tracking, R, probe authority) | **no.** It is capability at that checkpoint (`p_*`), never an event time |
| `derived` | earlier outputs in the run dir, e.g. `a_checkpoint_ladder.json` (checkpoint replay, 50–300), `a_credit_audit.json`, `g1_evaluation.json` | no, derived from replay |
| `online_rollout` | **none.** No training-time trajectories, gait, contacts or tracking were saved | — |

Naming rule: online quantities are `h_*`, replay quantities `p_*`. "Onset"
is used only for `h_*`. Probe-derived times are `t_probe-*`. Probe authority
(`p_authority`) and logged authority (`h_preference_authority`) are never
plotted as one series.

`h_preference_authority` is the online median action change between the
rollout preference and a fresh sampled preference, on 1024 rollout states
(`train_v4c.py`, lines 156–161 of the current file; that block is unchanged since `822cbcb`). `h_reward_per_step` is the mean over all
sampled preferences and cardinalities, not a per-preference value.

## Temporal resolution

- Online: every iteration (98 304 samples).
- Replay: every 50 iterations only. No capability claim can be finer than 50
  iterations without new checkpoints, which would need retraining. Wording:
  "no precursor observed at available resolution", never "no precursor
  existed".

## Preference exposure

Evidence level: **1, contract-level only.**

- Level 2 (reconstructed) is not available. The sampler redraws
  preferences for done envs at every reset (`train_v4c.py:137`, current file) from a
  generator it shares with minibatch permutation (`update`) and the
  authority probe. The realized sequence therefore depends on the done
  pattern, which depends on GPU physics. Reconstructing it needs a
  bit-exact training replay, which was not tested and is unlikely under
  PhysX GPU nondeterminism.
- Level 3 (observed) is not available. Sampled preferences were not logged.

Allowed claim: "no designed difference in preference exposure between
seeds". Not allowed: "the seeds received equivalent preference sequences",
or any claim that s73102 happened to see more T-heavy sets.

## Controls

| tier | runs | same code as target | note |
|---|---|---|---|
| matched batch | V4-C G1-2 s73101, s73103 (same fold); G1-3 s73101–73103 (other fold, trained {2, 4}) | yes (`822cbcb`, pre-fix normalizer) | primary controls |
| extended | V4-C G1-2 / G1-3 s73104–73108 | no: post-`61a483d` normalizer (declared difference, `f5f861d`) | characterization only, flagged |

The other two partial behaviors (V4-C G1-2 s73104 T⁺, V4-C3 G1-1 s73103 T⁺)
are T⁺-only and come from different code or lineage. They are secondary
reference points, not controls.

## Excluded (cross-lineage, same seed number)

Never compared with the target as if they were the same run:

- `runs/objective_set_g1_{2,3}_seed73102-2026-09-25/` — old V3 G1 lineage
  (different script, hash-frozen manifest, 8-env fine-tune, checkpoints
  model_0/10/20/30);
- `runs/teacher_v4_c2s1_g1_{2,3}_seed73102-2026-09-27/` — V4-C2S-R1 (S1
  objective);
- `runs/teacher_v4_c3_g1_{1,2}_seed73102-2026-09-28/` — V4-C3 (K = 3,
  dead-critic gate, post-fix normalizer).

## Next

F2-A (matched-seed bifurcation) needs its own frozen contract: the
definitions of `h_*` and `p_*`, the reference t₀ for ΔT/ΔD/ΔO, the
objective-space trajectories (T vs |D|, T vs |O|) as primary display, with
ratios only alongside their numerator and denominator.
