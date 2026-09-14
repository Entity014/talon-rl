# talon-rl — working notes for whoever (human or agent) touches this repo

This is a prelim/smoke-test scaffold, not a trained system — read
[README.md](README.md) and [docs/mdp.md](docs/mdp.md) first for scope and
known gaps before changing anything below.

## Conventions

(Borrowed from [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)'s
CONTRIBUTING.md — a robotics codebase with the same "small research/control repo, non-obvious
invariants" shape as this one.)

- **Comments say why, not what.** The reason a thing is the way it is outlives the code — see the
  invariants below for the standard this repo already holds itself to.
- **Commit messages are terse and why-focused.** State the reason for the change, not a restated
  diff; a scope prefix (`moppo:`, `reward:`, `preference:`) is welcome when it disambiguates, but
  isn't mandatory.
- **Every non-obvious decision gets a test, and the test's comment says which failure it exists to
  prevent.** Especially true here since there's no oracle to check RL correctness against yet —
  the test suite is what stands in for one.
- **Reach for an existing library before writing it yourself.** Dependency count is not what's
  being optimized; maintenance is.

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
- **`BaseTalonEnv`'s contract is batch-native.** Every `reset()`/`step()`
  value has a leading `(num_envs, ...)` axis, no exceptions — a new env
  that returns a single-env-shaped value (or squeezes/broadcasts away the
  leading axis) will silently break `moppo.py`'s vectorized rollout math
  instead of raising. See `talon_rl/envs/base_env.py`'s docstring.

## Before claiming something works

Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/` — 23 passed + 1 skipped
as of this writing (reward terms, preference math, end-to-end smoke tests on
both the dummy env and, when Isaac Sim is installed, the real Isaac Lab env;
`test_a1_env.py` is GPU/Isaac-Sim-gated and skips on this repo's default
3.12 .venv). A green suite proves the wiring, not correctness of the RL
algorithm's convergence behavior; there's no oracle to check against until
there's a real terrain/env to train on.
