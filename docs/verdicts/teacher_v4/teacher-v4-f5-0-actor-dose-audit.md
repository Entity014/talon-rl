# Teacher V4 — F5-0 Actor/Exploration Dose Audit (read-only)

Status: **DESCRIPTIVE, 2026-09-28.** Online logs only (`h_*`). Nothing trained. It calibrates the direction and size of F5 treatments before the F5 contract is frozen.
Output: `runs/teacher_v4_f5_0_dose_audit-2026-09-28.json`

Window means (iterations). V4-C runs are K = 4 with the pre-fix normalizer.
B0 is F3 / F4 (K = 3, post-fix). They are not directly comparable, and are
listed only for levels.

| run | log_std 201–300 | clip_frac 20–40 | KL 20–40 | LR 20–40 | LR 41–100 | LR 201–300 |
|---|---|---|---|---|---|---|
| **V4-C G1-2 s73102 (walker)** | **−1.57** | **0.197** | **0.0140** | **4.6e-3** | **1.5e-3** | **4.0e-4** |
| V4-C G1-2 s73101 | −1.37 | 0.170 | 0.0126 | 3.0e-3 | 1.1e-3 | 3.0e-4 |
| V4-C G1-2 s73103 | −1.17 | 0.151 | 0.0121 | 1.9e-3 | 0.8e-3 | 3.0e-4 |
| V4-C G1-3 s73101–73103 | −1.23 to −1.26 | 0.173–0.183 | 0.0128–0.0136 | 2.0–2.8e-3 | 0.8–1.1e-3 | 3.0–4.0e-4 |
| F3/F4 B0 s74101–74103 | −1.04 to −1.24 (−1.30 to −1.55 at 451–600) | 0.163–0.175 | 0.0129–0.0135 | 2.1–3.0e-3 | 0.8–1.1e-3 | 3.0–4.0e-4 |

Clip fraction at iterations 26–40 (where F2-A found the only specific
onset): the walker sits at 0.17–0.23, the two matched controls at 0.11–0.21.
**The sign is higher.**

## Reading

- **Actor-update geometry: the walker made larger updates.** Its clip
  fraction and KL were higher early, and its adaptive LR stayed about
  1.3–2.4× the matched controls' through iteration 100 (1.3× at 201–300).
  This is the direction of the one specific F2-A signal.
- **Stochasticity: the walker was less stochastic, not more.** Its log_std
  fell fastest (−1.57 against −1.17 / −1.37). F2-A found log_std and entropy
  nonspecific (q 0.77). The evidence does not support "preserve entropy or
  log_std" as the mechanism. If anything it points the other way, weakly.
- Magnitudes: walker-to-control ratios are about 1.1–1.2× for KL and about
  1.3–2.4× for LR. The adaptive rule targets desired_kl = 0.01, so a larger
  desired_kl is the knob that moves LR, KL and clip fraction together.

Descriptive and based on one walker. It sets the direction of F5 arms and
does not show a mechanism.
