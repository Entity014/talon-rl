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
  order.** Every place that builds the Phase-1 5-vector (reward.py, preference.py,
  moppo.py) indexes by name via this tuple, never a hardcoded position. If
  you add a 6th term, add it to `term_names` + `active` + `_TERM_FUNCS` in
  `reward.py` together — nowhere else needs to change.
- **$w$ always sums to 1 and never violates the impact/balance/progress
  floors** (`RewardVectorCfg.impact_floor_eps`/`balance_floor_eps`/
  `progress_floor_eps`). Phase-1 MOPPO produces it through
  `preference.sample_preference_vector` -> `MOPPOTrainer._clip_preference`
  (`preference.floor_clip_terms`) once per episode. A future HLP/manual
  scheduler additionally applies `preference.rate_limit` before the floor
  clip. Don't construct or mutate a preference vector by hand anywhere
  else — it's the one invariant the Multi-Objective Module depends on.
  `progress_floor_eps` (added 2026-09-19) exists because a same-day
  preference-curriculum experiment that tried to fix progress starvation
  via temporary Dirichlet-alpha annealing made things worse once the
  anneal faded back to uniform — a permanent floor was the untried
  alternative (see talon-thesis/03_Daily_Notes/2026-09-19.md).
- **`BaseTalonEnv.obs_dim` excludes $w$.** The preference vector is appended
  to the observation inside `scripts/rl/core/algorithms/moppo.py`, not by the environment. Any
  new env (including the eventual Isaac Lab one) must NOT put $w$ into its
  own `obs` array — it'll get double-appended and silently break the
  policy's input shape.
- **A `transition` dict must carry every active reward input**:
  `v_actual`, `v_command`, `joint_torque`, `joint_vel`,
  `foot_contact_force`, `action`, `prev_action`, `joint_acc`, `roll_pitch`,
  plus `obs`. `v_z`/`height` (balance's vertical-bounce/crouch
  sub-penalties), `foot_vel`/`undesired_contact_count` (impact's
  foot-slip/undesired-contact sub-penalties), `foot_air_time_reward`
  (progress's positive step-completion bonus), and
  `hip_qdot_L`/`hip_qdot_R`/`hip_q_L`/`hip_q_R` (balance's
  hip_activation/hip_sym sub-terms, 2026-09-20 — mean joint
  velocity/position per side's hip pair, real physics not a commanded-
  target proxy) are all OPTIONAL
  (`dict.get(...)`, default `None` = no contribution) — see
  docs/mdp.md's "Grouped sub-penalties" note — so envs without real
  vertical/foot kinematics (e.g. `DummyTalonEnv`) don't need placeholder
  values for them. `foot_air_time_reward` is STATEFUL (needs the previous
  step's contact to detect a touchdown event) — only `IsaacLabTalonEnv`
  computes it (see `_compute_foot_air_time_reward`'s docstring for the
  double-call-per-step guard it relies on); a new env wanting this term
  needs its own equivalent per-lane, per-foot bookkeeping, not a stateless
  one-liner like the other optional fields. If
  you write a new env, grep `reward.py` for the exact key names before
  assuming the shape is obvious. (A second reward module now exists for the
  TienKung sibling module, with its own key set — see README's "A separate
  module" section.)
- **Running per-objective reward normalization lives in
  `RunningMeanStd` (`scripts/rl/core/running_norm.py`)**, wired into
  `MOPPOTrainer._collect_rollout` — it divides each term by its own running
  std (no mean-centering) before GAE, so `smoothness` no longer dominates
  `progress`'s gradient despite ~1000x raw-scale gap (see docs/mdp.md). The
  normalizer's stats round-trip through `save()`/`load()`'s checkpoint —
  don't reset them on resume, that would shock the reward scale the value
  function was trained against.
- **`DummyTalonEnv` numbers mean nothing about locomotion.** If a change
  makes `train_prelim.py`'s printed rewards go up, that is not evidence the
  MOPPO implementation got better at anything real — it only means the loop
  still runs. Don't cite dummy-env results anywhere in the thesis.
- **`BaseTalonEnv`'s contract is batch-native.** Every `reset()`/`step()`
  value has a leading `(num_envs, ...)` axis, no exceptions — a new env
  that returns a single-env-shaped value (or squeezes/broadcasts away the
  leading axis) will silently break `moppo.py`'s vectorized rollout math
  instead of raising. See `talon_rl/envs/base_env.py`'s docstring.

## scripts/rl layout

The one-shot experiment scripts live under `scripts/rl/experiments/<line>/`, one
directory per experiment line, with `experiments/shared/` holding the modules
other scripts import. Only `train_prelim.py`, `play.py`, `sim2sim.py`,
`diagnostics.py` and `launch_isaac_lab_via_pytest.py` sit at the top of
`scripts/rl/`, beside `core/`.

- **A script two levels down finds the repo root at
  `Path(__file__).resolve().parents[4]`**, not `parents[2]`. The scripts in
  `experiments/` that import a sibling do it absolutely, as
  `rl.experiments.<line>.<module>`, and carry a `sys.path` bootstrap at the top
  so that resolves when the file is run directly.
- **Anything that writes into a run directory should subclass
  `rl.core.run_report.RunReport`** — `Freeze` for the two files that close an
  experiment out, `IsaacAudit` for a rollout audit, `OfflineAudit` for one that
  reads a finished run. They all take `--out`, which is what makes a script
  re-runnable for comparison: `runs/` is gitignored, so re-running one in place
  destroys the only copy of what it wrote.
- **`git tag pre-reorg-2026-09-26` holds `scripts/rl` exactly as it stood before
  that move.** The `PROVENANCE_MANIFEST.json` files in `runs/` pin sha256 values
  of the scripts that produced each artifact, under their old flat paths, and
  the move rewrote 593 files. 108 of the 133 recorded Python source hashes
  resolve against that tag and none against `main`, so check a recorded hash
  there. 502 of the files in it were never tracked anywhere else.

## Before claiming something works

Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ --ignore=tests/test_sim2sim.py`
with `~/isaac-lab-env/bin/python`, and separately
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_sim2sim.py` with
`.venv/bin/python` (only that venv has `mujoco` installed) — 144 + 12
passed + 1 pre-existing unrelated failure as of this writing (reward
terms, preference math, end-to-end smoke tests on both the dummy env and
the sim2sim MuJoCo path). `test_a1_env.py` is GPU/Isaac-Sim-gated and, on
this machine, silently kills the whole pytest process on collection
(unrelated Isaac Sim quirk, not a regression) — excluded above; run it
separately and individually when actually testing real Isaac Lab env
wiring. A green suite proves the wiring, not correctness of the RL
algorithm's convergence behavior; there's no oracle to check against until
there's a real terrain/env to train on.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
