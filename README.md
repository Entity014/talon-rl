# talon-rl

Training code for **TALON** (*Terrain-Adaptive Locomotion via Objective
Negotiation*) — the Multi-Objective Module prelim, split out of the
[TALON-thesis](https://github.com/Entity014/TALON-thesis) proposal repo.

## Status: prelim only, not a trained/converged result

This repo does **not** produce a trained policy or a locomotion result. It
now has two envs implementing the same [`BaseTalonEnv`](talon_rl/envs/base_env.py)
contract: a physics-free `DummyTalonEnv` for fast CPU iteration, and a real,
vectorized Isaac Lab environment (`IsaacLabTalonEnv`, registered as
`Isaac-Talon-A1-v0`) that runs thousands of parallel Unitree A1 clones on a
GPU machine via `ManagerBasedRLEnv`. Both are smoke tests proving the RL loop
is wired correctly (`obs → policy(w) → action → reward vector → vector critic
→ PPO update`), not a locomotion result. Nothing about "the policy improved"
on either env is meaningful; only "the pipeline runs without breaking" is.

**Isaac Sim is the target simulator, and it's now wired up.** `IsaacLabTalonEnv`
implements the same `reset()`/`step()` contract as `DummyTalonEnv` —
`scripts/rl/core/algorithms/moppo.py` doesn't need to change at all to switch between them;
see `--env dummy` vs `--env isaac_lab` below.

See [docs/mdp.md](docs/mdp.md) for the field-by-field rationale behind every
observation/action/reward-vector entry, and [CLAUDE.md](CLAUDE.md) for the
invariants this code depends on before you change anything.

## Scope of this prelim (see chapter3.tex for the full pipeline)

**In scope:**

- Multi-Objective Module: MOPPO, Dirichlet-sampled preference vector $w$,
  rate-limiter + floor-clip, vector critic $V(s,c,w)$ (table in §3.2.3)
- All 5 reward-vector terms from table 3.3: Progress, Clearance, Energy,
  Impact, Smoothness

**Out of scope (separate milestones, after the proposal defense):**

- Adaptation Module ($\hat z_t, \sigma_t$ — payload/morphology awareness)
- Exteroception Module (depth-camera terrain/obstacle perception)
- Full terrain curriculum (§3.3.1) — the qualitative gap/chasm test scenario
  from §3.7.2 is the eventual target, not yet built here
- Real Unitree A1 hardware (the Isaac Lab/Isaac Sim *simulated* A1 environment
  exists — see Status above — but nothing here has run on the physical robot)

## Known gaps vs. chapter3.tex (don't mistake this for the real thing)

- No OOD monitor gating $w$ (depends on $\sigma_t$ from the Adaptation Module,
  which is out of scope).
- **Terminal-step reward/action mispairing under auto-reset.** For a lane
  that terminates on a given step, the reward vector for that step is
  computed from the *next* episode's first frame (reset()'s fresh values —
  e.g. zeroed `action`/`prev_action`, a freshly-randomized `obstacle_dist`),
  not from the actual terminal frame/action that caused the termination.
  This is consistent across both envs (so `--env dummy`/`--env isaac_lab`
  stay a true drop-in swap) and gives roughly 1-in-`horizon` steps slightly
  wrong credit assignment. Deliberately out of scope for this prelim — see
  [`talon_rl/envs/base_env.py`](talon_rl/envs/base_env.py) for the exact
  mechanism.

## A separate module: TienKung bimanual box-carry (not part of this thesis)

`talon_rl/tasks/manipulation/tienkung_env/` and `talon_rl/assets/tienkung2_lite/`
are a **separate research application** — applying this thesis's
Multi-Objective RMA methodology to a different robot (TienKung2 Lite, a
bipedal humanoid) and a different task (bimanual box-carrying, not
locomotion). This is explicitly **out of TALON's own defended scope**:
`00_Proposal §3.6` disclaims cross-embodiment transfer across different
joint topologies, and TienKung's joint topology is nothing like the A1's.

It lives in this repo as a sibling module (not a separate repo) because it
reuses this repo's generic infrastructure directly — `BaseTalonEnv`, the
dim-agnostic preference-sampling functions in `scripts/rl/core/preference.py`
— rather than because it's part of the thesis's claimed contribution. See
`docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md` for the
full design rationale. Round 1 only: asset vendoring + MDP config/reward +
a physics-free dummy env, no real Isaac Lab env yet.

## Layout

Split the same way as [jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot):
`talon_rl/` is the portable, pip-installable task package (env/asset/MDP
definitions only — swappable to any training algorithm); the RL algorithm
itself is driver code under `scripts/`, not part of the installed package.

```text
talon_rl/
  config.py          # ObservationSpaceCfg / ActionSpaceCfg / RewardVectorCfg / PreferenceCfg
                      # — mirrors chapter3.tex tables 3.1-3.3
  reward.py          # the 5 reward-vector terms + compute_reward_vector()
  envs/
    base_env.py        # interface any env (real or dummy) must implement
  assets/
    unitree_a1/
      a1.py               # TALON_A1_CFG — UNITREE_A1_CFG + RMA Kp/Kd, local usd_path
    data/Robots/unitree_a1/ # vendored A1 USD/mesh/texture + UrdfConverter config.yaml (~42MB, no live Nucleus dependency)
    config/
      extension.toml      # Isaac Sim Kit extension descriptor for talon_rl.assets
  tasks/locomotion/a1_env/
    a1_env.py            # IsaacLabTalonEnv(ManagerBasedRLEnv, BaseTalonEnv), registered Isaac-Talon-A1-v0
    a1_env_cfg.py         # scene/observations/actions/terminations/events manager configs
    mdp/                   # scripted MDP term functions (observations.py, terminations.py)
tests/                # pytest — reward terms, preference math, end-to-end smoke tests
scripts/
  rl/                   # driver code, matches jaykorea's own scripts/co_rl/ split exactly:
                        # entry point at this level, library nested one level deeper in core/
                        # (core/algorithms/ — one file per algorithm, e.g. multiple SAC/TQC-
                        # style variants — vs. this repo's current single MOPPO algorithm)
    train_prelim.py      # entry point — mirrors co_rl/train.py sitting beside core/
    play.py                # loads a checkpoint, runs deterministic inference,
                           # optionally exports (--export) and/or analyzes (--analyze/--plot)
    sim2sim.py               # mechanism-only Isaac Sim -> MuJoCo policy rollout (00_Proposal §3.4)
    core/
      algorithms/
        moppo.py            # MOPPOConfig + MOPPOTrainer — preference-conditioned PPO
                            # (vector critic, w . advantage); rollout collection stays
                            # here rather than a shared runner since the per-step
                            # preference-vector resampling is MOPPO-specific, not generic
      modules/
        actor_critic.py      # ActorCritic network shape — reusable across algorithms
      storage/
        rollout_storage.py    # gae_per_objective — GAE math, reusable across algorithms
      wrapper/
        exporter.py            # TorchScript policy export for deployment/sim2sim
      run_dir.py                 # logs/talon_rl/<run>/ management — config.yaml dump,
                                 # checkpoint.pt, --load_run "last" resolution
      analyzer.py                 # play.py's --analyze/--plot: per-step signal
                                  # collection + matplotlib PNGs, own impl (not a port —
                                  # see the module's own docstring for why)
      sim2sim.py                   # A1 MuJoCo observation-building + rollout mechanism
      preference.py                  # Dirichlet sampling, rate-limiter, floor-clip
      obs_stack.py                     # batched (N, stacks, obs_dim) actor/critic observation history
      dummy_env.py                       # physics-free smoke-test env (BaseTalonEnv-implementing, training-only)
```

```text
talon_rl/tasks/manipulation/tienkung_env/  # separate module, see the section above — not part of this thesis
  config.py             # ObservationSpaceCfg / ActionSpaceCfg / RewardVectorCfg for this task
  reward.py             # 5-term reward vector, retargeted from talon_rl/reward.py's shape
  dummy_env.py           # physics-free smoke-test env, mirrors scripts/rl/core/dummy_env.py's structure
talon_rl/assets/tienkung2_lite/  # vendored TienKung2 Lite asset, mirrors assets/unitree_a1/'s pattern
```

## Running it

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
python scripts/rl/train_prelim.py --updates 50
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` works around an unrelated ROS
`launch_testing` pytest plugin conflict on some machines — harmless to
include everywhere.)

### Isaac Lab smoke test (requires the separate `~/isaac-lab-env` venv, GPU machine only)

```bash
source ~/isaac-lab-env/bin/activate
python scripts/rl/train_prelim.py --env isaac_lab --updates 5 --num_envs 4096
```

Proves the pipeline runs against a real, vectorized Isaac Lab environment
(`ManagerBasedRLEnv`, 4096 parallel A1 clones) — same "doesn't crash" bar as
the single-env 2026-09-13 version, not a locomotion result. See
`docs/superpowers/specs/2026-09-14-vectorized-isaac-lab-env-design.md` for
the vectorization design and
`docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md` for the
original single-env setup and known limitations.

## Next milestones (after proposal defense)

1. ~~Write `IsaacLabTalonEnv(BaseTalonEnv)` against the real Unitree A1 asset.~~ Done — see `talon_rl/tasks/locomotion/a1_env/`.
2. ~~Add running per-objective reward normalization.~~ Done — see `RunningMeanStd` in `scripts/rl/core/running_norm.py`, wired into `MOPPOTrainer._collect_rollout`.
3. Reproduce the RMA (Kumar et al. 2021) two-phase teacher-student baseline —
   this is the Adaptation Module, currently entirely absent here.
4. Build the Exteroception Module (depth → terrain/obstacle embedding).
5. Run the §3.7.2 qualitative gap/chasm test for real, with a real reward
   trade-off, not the dummy env's toy one.
