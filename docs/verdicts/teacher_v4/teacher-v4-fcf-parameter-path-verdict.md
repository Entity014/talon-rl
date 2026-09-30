# Teacher V4 — FC-F parameter-path attribution verdict

Status: **RESULT, 2026-10-01.** The [FC-F contract](../../contracts/teacher_v4/teacher-v4-fcf-parameter-path-contract.md) was frozen at `d0c3929` before any FC-F branch outcome. No arms, thresholds, or repeats were added after results. All 46 branches trained and replayed; all 36 registered paired comparisons are valid. Output: `runs/teacher_v4_fc-2026-09-30/fcf/` (`fcf_stage1.json`, `expand.txt`, `fcf_stage2.json`, per-branch metrics and replay traces). The live-rollout preflight is in `runs/teacher_v4_fc-2026-09-30/fcf_preflight/seed79101/live_audit.json`.

The primary effect is `matched Δ_R(FULL) − matched Δ_R(routed arm)` in percentage points. Positive means blocking a task-gradient route lowered the harmful R separation change. Raw effects are descriptive. Stage 1 used fresh repeats r = 6–8; only six open seed × arm contrasts expanded to r = 9–10. The existing PPO scalarization and advantage rule were held fixed; only the new task-stream actor-gradient route changed.

| Seed | Routed arm | Stage | Matched paired effects, pp | Median matched / raw, pp | Registered class |
|---|---|---:|---|---:|---|
| 79101 | NO-PREF-PATH | 1 | +13.0, +14.3, +10.7 | +13.0 / +13.0 | replicated attenuation |
| 79101 | NO-BASES | 2 | +9.5, +3.9, +29.6, +12.0, +10.5 | +10.5 / +9.4 | replicated attenuation |
| 79101 | SHARED-FEATURES-ONLY | 1 | +12.3, +5.0, +20.6 | +12.3 / +10.2 | replicated attenuation |
| 79101 | PREF-ONLY | 2 | +8.3, +9.8, +18.5, +8.9, +15.0 | +9.8 / +13.5 | not resolved by registered budget |
| 79103 | NO-PREF-PATH | 2 | +8.0, +10.5, +8.8, +8.7, −0.8 | +8.7 / +6.7 | not resolved by registered budget |
| 79103 | NO-BASES | 2 | +11.4, +3.6, +2.6, +4.9, +2.0 | +3.6 / +6.1 | within practical band |
| 79103 | SHARED-FEATURES-ONLY | 2 | +6.1, −2.9, +11.2, −3.7, −11.0 | −2.9 / −0.3 | not resolved by registered budget |
| 79103 | PREF-ONLY | 2 | +6.2, −0.1, +6.6, +2.2, +2.2 | +2.2 / +1.0 | within practical band |

The table rounds values; the script applies strict thresholds to unrounded effects. Every pair had two viable endpoints and a matched value. The FULL matched Δ_R medians were +11.6% in 79101 and +6.5% in 79103 over r = 6–10. Binary inversion labels varied and did not vote.

## Registered reading

**No task-gradient parameter route is established across both primary seeds.** In 79101, blocking G4–G6 (NO-PREF-PATH), blocking G3–G4 (NO-BASES), or allowing task gradient only into G1–G2/G7 (SHARED-FEATURES-ONLY) all replicate attenuation relative to FULL. This localizes an important task-route effect in that seed, but the overlapping masks do not identify a single group: G4 is blocked in all three, yet interactions with G3/G5/G6 and the optimizer remain possible. PREF-ONLY in 79101 stayed below the strict +10 pp median threshold and is unresolved.

In 79103 no mask met the registered attenuation rule. NO-BASES and PREF-ONLY fell within the predefined practical band; that class is descriptive, not formal equivalence or proof of no effect. NO-PREF-PATH had a +8.7 pp median and is unresolved, so treating it as a null would be incorrect. SHARED-FEATURES-ONLY changed sign across repeats and is unresolved. Thus the route response is **seed-dependent under this continuation**, while FC-E's broader task-pressure *loss-formulation* effect remains replicated.

## Optimizer and interpretation limits

The live-rollout preflight used a real 98,304-row rollout and audited a 4,096-row minibatch. FULL matched the legacy path after the optimizer step (maximum parameter and Adam-state differences 0); per-route reconstructed post-clip gradients differed from the actual optimizer gradients by at most 6×10⁻⁸. All G1–G7 had nonzero unmasked task gradients in that minibatch; the critic had no actor gradient.

Routing changed global actor clipping. Across repeats, the mean clip coefficient was 0.405 for FULL versus 0.300 for NO-PREF-PATH in 79101, and 0.276 versus 0.196 in 79103. NO-BASES averaged 0.463 and 0.337 respectively. Since global clipping scales the entire actor gradient, R/O's **post-clip** contribution can change even when its pre-clip loss and gradient calculation are held fixed. Adam state from u0 also retains historical task influence. The result is attribution of task-gradient **routing within the current PPO training stack**, not a pure independent vector-path effect.

The registered budget is exhausted; there is no Stage 3. These results do not yet justify choosing objective-specific adapters alone, a shared-path projection alone, or a hybrid as the final architecture. Any finer G3/G4/G5/G6 split or optimizer-factor intervention needs a new predeclared contract.
