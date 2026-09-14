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
`training/moppo.py` doesn't need to change at all to switch between them; see
`--env dummy` vs `--env isaac_lab` below.

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

- **No running per-objective reward normalization.** chapter3.tex explicitly
  calls this out as necessary ("ป้องกันไม่ให้เทอมที่มีขนาดใหญ่ครอบงำเกรเดียนต์ของเทอมอื่น") —
  it's skipped here, and you can see why in the smoke-test output: the
  `smoothness` term sits around -300 while `progress` sits around 0-1, which
  will dominate the vector critic's loss until normalization is added.
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

## Layout

```text
talon_rl/
  config.py          # ObservationSpaceCfg / ActionSpaceCfg / RewardVectorCfg / PreferenceCfg
                      # — mirrors chapter3.tex tables 3.1-3.3
  reward.py          # the 5 reward-vector terms + compute_reward_vector()
  preference.py       # Dirichlet sampling, rate-limiter, floor-clip
  obs_stack.py         # batched (N, stacks, obs_dim) actor/critic observation history
  envs/
    base_env.py        # interface a real Isaac Lab env must implement
    dummy_env.py        # physics-free smoke-test env
  assets/
    a1.py               # Unitree A1 Isaac Lab asset config
  tasks/locomotion/a1_env/
    a1_env.py            # IsaacLabTalonEnv(ManagerBasedRLEnv, BaseTalonEnv), registered Isaac-Talon-A1-v0
    a1_env_cfg.py         # scene/observations/actions/terminations/events manager configs
    mdp/                   # scripted MDP term functions (observations.py, terminations.py)
  training/
    moppo.py           # preference-conditioned PPO (vector critic, w . advantage)
tests/                # pytest — reward terms, preference math, end-to-end smoke tests
scripts/
  train_prelim.py      # entry point
```

## Running it

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
python scripts/train_prelim.py --updates 50
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` works around an unrelated ROS
`launch_testing` pytest plugin conflict on some machines — harmless to
include everywhere.)

### Isaac Lab smoke test (requires the separate `~/isaac-lab-env` venv, GPU machine only)

```bash
source ~/isaac-lab-env/bin/activate
python scripts/train_prelim.py --env isaac_lab --updates 5 --num_envs 4096
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
2. Add running per-objective reward normalization.
3. Reproduce the RMA (Kumar et al. 2021) two-phase teacher-student baseline —
   this is the Adaptation Module, currently entirely absent here.
4. Build the Exteroception Module (depth → terrain/obstacle embedding).
5. Run the §3.7.2 qualitative gap/chasm test for real, with a real reward
   trade-off, not the dummy env's toy one.
