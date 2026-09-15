# MDP definition

Ported from `chapters/chapter3.tex` tables 3.1-3.3 in
[TALON-thesis](https://github.com/Entity014/TALON-thesis) — this file exists
so the rationale for each field lives next to the code that implements it,
not only in the LaTeX source.

## Observation (`config.ObservationSpaceCfg`)

| Field | Dim | Source | Why it's here |
|---|---|---|---|
| `joint_pos` | 12 | proprioceptive | RMA `x_t` component \cite{kumar2021} |
| `joint_vel` | 12 | proprioceptive | RMA `x_t` component |
| `roll_pitch` | 2 | IMU | RMA `x_t` component |
| `foot_contact` | 4 | proprioceptive, binarized | RMA `x_t` component |
| `prev_action` | 12 | feedback | closes the loop for the policy |
| `command` | 3 | external ($v_x, v_y, \omega_z$) | what "progress" is tracking against |
| `preference` ($w$) | 5 | Multi-Objective Module | appended in `training/moppo.py`, **not** by the env — see `envs/base_env.py` |

**Deliberately absent right now**: $\hat z_t, \sigma_t$ (Adaptation Module)
and the Exteroception embedding. Both are `[TBD]` in the thesis pending
ablation, and out of scope for the Multi-Objective Module prelim — adding
them later just means widening `ObservationSpaceCfg` and updating
`BaseTalonEnv`'s transition dict contract, not touching `moppo.py`.

## Action (`config.ActionSpaceCfg`)

$a_t \in \mathbb{R}^{12}$ — target joint angle, converted to torque downstream
by a PD controller: $\tau = K_p(a_t - q) + K_d(\dot a_t - \dot q)$
\cite{kumar2021}. This repo doesn't implement the PD conversion itself — a
real env (Isaac Lab or the physics-free dummy) owns that.

## Reward vector (`config.RewardVectorCfg`, `reward.py`)

Fixed order — **never** reorder `term_names` without also checking every
place that indexes by position (there currently isn't one; everything goes
through the name, but a raw index would silently break if this changes):

1. `progress` — exp-kernel velocity tracking (go-anywhere navigation)
2. `clearance` — obstacle negotiation (scripted signal in the dummy env;
   real signal comes from the Exteroception Module, not built yet)
3. `energy` — negative raw power (efficiency)
4. `impact` — continuous impact mitigation \cite{strauch2025crashcourse},
   penalizes peak contact force above a threshold, not just falls
5. `smoothness` — fixed-weight regularizer (action rate + joint accel), **not**
   part of the preference vector $w$ in the real system — see below

### Running per-objective normalization

chapter3.tex is explicit that every term needs "การปรับมาตรฐานแบบเคลื่อนที่ต่อ
วัตถุประสงค์ (running per-objective normalization) ... เพื่อป้องกันไม่ให้เทอมที่มี
ขนาดใหญ่ครอบงำเกรเดียนต์ของเทอมอื่น" — `RunningMeanStd`
(`scripts/rl/core/running_norm.py`, Welford batched update) tracks a running
mean/std per reward term and `MOPPOTrainer._collect_rollout` divides each
term by its running std (no mean-centering, so a bounded term like
`clearance` keeps its 0-boundary meaning) before GAE. Raw scale is still
wildly uneven — `smoothness` around -300 vs. `progress` around 0-1 — but the
stored/normalized reward stays within the normalizer's clip range
(`±10` by default) regardless.

## Preference vector $w$ (`preference.py`)

$w \sim \mathrm{Dirichlet}(\alpha{=}1.0)$ per episode (fig. 3.3), then every
step: rate-limited (`max_delta_per_step`, caps how fast $w$ can drift) ->
floor-clipped ($w_{impact} \geq \varepsilon$, chapter3.tex §3.2.3 — impact
mitigation must never be fully zeroed out) -> renormalized to sum to 1.

The 5th reward term (`smoothness`) is summed with a fixed small constant
weight in the real system, not driven by $w$ — this repo currently folds it
into the same 5-dim $w$ for simplicity (see `reward.py` module docstring).
Splitting it out is a small follow-up, not a design change: give
`RewardVectorCfg` a separate `fixed_weights` dict for `smoothness`, drop it
from `preference_dim`, and update `moppo.py`'s advantage scalarization to
`w . adv[:4] + fixed_weight * adv[4]`.
