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
| `base_ang_vel` | 3 | IMU (gyro rate) | added 2026-09-18 — neither RMA's `x_t` nor this repo previously observed trunk angular velocity at all, only individual `joint_vel` and a static `roll_pitch` snapshot; no signal for how fast the trunk itself is rotating. `root_ang_vel_b` was already read in `reward.py`'s `v_actual` but never exposed to the policy. Confirmed missing by diffing against jaykorea/Isaac-RL-Two-wheel-Legged-Bot's own quadruped env, which observes it as standard practice |
| `projected_gravity` | 3 | IMU-derived (gravity direction in body frame) | added 2026-09-18 alongside `base_ang_vel` — standard complement to `roll_pitch` in legged-gym-style observations, a bounded unit vector with no Euler-angle singularity |
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
2. `energy` — negative raw power (efficiency)
3. `impact` — continuous impact mitigation \cite{strauch2025crashcourse},
   penalizes peak contact force above a threshold, not just falls
4. `smoothness` — action-rate and joint-acceleration penalty (hardware wear)
5. `balance` — negative squared trunk roll/pitch; a dense anti-fall signal
   before the contact-based fall termination fires

All five are dimensions of the Phase-1 preference vector $w$. This follows
AMOR's use of smoothness and root orientation as separately weighted
objectives: the policy can negotiate tracking, wear, and stability rather
than hard-coding one simulator-specific trade-off. Fall termination and the
eventual hardware safety supervisor remain independent hard safety layers.

`clearance` is intentionally absent for now. Its current scripted obstacle
distance is not a meaningful training signal; putting a zero-valued channel
in $w$ would train policy contexts that cannot receive a corresponding reward.
It becomes a sixth objective only with the Exteroception Module.

### Running per-objective normalization

chapter3.tex is explicit that every term needs "การปรับมาตรฐานแบบเคลื่อนที่ต่อ
วัตถุประสงค์ (running per-objective normalization) ... เพื่อป้องกันไม่ให้เทอมที่มี
ขนาดใหญ่ครอบงำเกรเดียนต์ของเทอมอื่น" — `RunningMeanStd`
(`scripts/rl/core/running_norm.py`, Welford batched update) tracks a running
mean/std per reward term and `MOPPOTrainer._collect_rollout` divides each
term by its running std (no mean-centering) before GAE. Raw scale is still
wildly uneven — `smoothness` around -300 vs. `progress` around 0-1 — but the
stored/normalized reward stays within the normalizer's clip range
(`±10` by default) regardless.

## Preference vector $w$ (`preference.py`)

$w \sim \mathrm{Dirichlet}(\alpha{=}1.0)$ once at each episode reset
(AMOR's Phase-1 protocol), then is floor-clipped
($w_{impact} \geq \varepsilon$, chapter3.tex §3.2.3 — impact mitigation must
never be fully zeroed out) and renormalized to sum to 1. It stays constant
until that lane's next reset.

A future HLP or manual hardware scheduler may emit $w_t$ during an episode;
only that deployment path applies `max_delta_per_step` as a rate limiter
before the same floor clip and the safety/OOD guards.
