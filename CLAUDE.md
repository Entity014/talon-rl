# talon-rl — working notes for whoever (human or agent) touches this repo

This is a prelim/smoke-test scaffold, not a trained system — read
[README.md](README.md) and [docs/mdp.md](docs/mdp.md) first for scope and
known gaps before changing anything below.

## Rules learned so far

- **`RewardVectorCfg.term_names` is the single source of truth for reward
  order.** Every place that builds a 5-vector (reward.py, preference.py,
  moppo.py) indexes by name via this tuple, never a hardcoded position. If
  you add a 6th term, add it to `term_names` + `active` + `_TERM_FUNCS` in
  `reward.py` together — nowhere else needs to change.
- **$w$ always sums to 1 and never violates the impact floor.** Always
  produce $w$ through `preference.sample_preference_vector` ->
  `preference.rate_limit` -> `preference.floor_clip`, in that order. Don't
  construct or mutate a preference vector by hand anywhere else — it's the
  one invariant the Multi-Objective Module depends on.
- **`BaseTalonEnv.obs_dim` excludes $w$.** The preference vector is appended
  to the observation inside `training/moppo.py`, not by the environment. Any
  new env (including the eventual Isaac Lab one) must NOT put $w$ into its
  own `obs` array — it'll get double-appended and silently break the
  policy's input shape.
- **A `transition` dict must carry every key `reward._TERM_FUNCS` expects**:
  `v_actual`, `v_command`, `obstacle_dist`, `joint_torque`, `joint_vel`,
  `foot_contact_force`, `action`, `prev_action`, `joint_acc`, plus `obs`. If
  you write a new env, grep `reward.py` for the exact key names before
  assuming the shape is obvious.
- **Don't compare reward-term magnitudes across terms until running
  per-objective normalization is added** (see docs/mdp.md). `smoothness`
  currently dominates `progress` by ~1000x in raw scale — this is a known,
  documented gap, not a bug to "fix" by reweighting `RewardVectorCfg`'s
  constants.
- **`DummyTalonEnv` numbers mean nothing about locomotion.** If a change
  makes `train_prelim.py`'s printed rewards go up, that is not evidence the
  MOPPO implementation got better at anything real — it only means the loop
  still runs. Don't cite dummy-env results anywhere in the thesis.

## Before claiming something works

Run `pytest tests/` — 12 tests as of this writing (reward terms, preference
math, one end-to-end smoke test on the dummy env). A green suite proves the
wiring, not correctness of the RL algorithm's convergence behavior; there's
no oracle to check against until there's a real terrain/env to train on.
