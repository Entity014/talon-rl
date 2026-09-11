# talon-rl

Training code for **TALON** (*Terrain-Adaptive Locomotion via Objective
Negotiation*) — the Multi-Objective Module prelim, split out of the
[TALON-thesis](https://github.com/Entity014/TALON-thesis) proposal repo.

## Status: prelim only, no Isaac Sim yet

This repo does **not** train on a real robot or simulator. There's no GPU in
the environment it was scaffolded in, so everything here runs on a
physics-free `DummyTalonEnv` — a smoke test proving the RL loop is wired
correctly (`obs → policy(w) → action → reward vector → vector critic → PPO
update`), not a locomotion result. Nothing about "the policy improved on the
dummy env" is meaningful; only "the pipeline runs without breaking" is.

**Isaac Sim is the target simulator once this moves to a GPU machine.** Swap
`DummyTalonEnv` for a real Isaac Lab environment that implements
[`BaseTalonEnv`](talon_rl/envs/base_env.py) — same `reset()`/`step()` contract
— and `training/moppo.py` doesn't need to change at all.

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
- The real Isaac Lab/Isaac Sim environment and Unitree A1 hardware

## Known gaps vs. chapter3.tex (don't mistake this for the real thing)

- **No running per-objective reward normalization.** chapter3.tex explicitly
  calls this out as necessary ("ป้องกันไม่ให้เทอมที่มีขนาดใหญ่ครอบงำเกรเดียนต์ของเทอมอื่น") —
  it's skipped here, and you can see why in the smoke-test output: the
  `smoothness` term sits around -300 while `progress` sits around 0-1, which
  will dominate the vector critic's loss until normalization is added.
- No OOD monitor gating $w$ (depends on $\sigma_t$ from the Adaptation Module,
  which is out of scope).
- Single dummy env, sequential rollout — not the vectorized multi-env
  collection a real Isaac Lab training run needs for throughput.

## Layout

```
talon_rl/
  config.py          # ObservationSpaceCfg / ActionSpaceCfg / RewardVectorCfg / PreferenceCfg
                      # — mirrors chapter3.tex tables 3.1-3.3
  reward.py          # the 5 reward-vector terms + compute_reward_vector()
  preference.py       # Dirichlet sampling, rate-limiter, floor-clip
  envs/
    base_env.py        # interface a real Isaac Lab env must implement
    dummy_env.py        # physics-free smoke-test env
  training/
    moppo.py           # preference-conditioned PPO (vector critic, w . advantage)
tests/                # pytest — reward terms, preference math, end-to-end smoke test
scripts/
  train_prelim.py      # entry point
```

## Running it

```bash
pip install -e ".[dev]"
pytest tests/
python scripts/train_prelim.py --updates 50
```

## Next milestones (after proposal defense)

1. Write `IsaacLabTalonEnv(BaseTalonEnv)` against the real Unitree A1 asset.
2. Add running per-objective reward normalization.
3. Reproduce the RMA (Kumar et al. 2021) two-phase teacher-student baseline —
   this is the Adaptation Module, currently entirely absent here.
4. Build the Exteroception Module (depth → terrain/obstacle embedding).
5. Run the §3.7.2 qualitative gap/chasm test for real, with a real reward
   trade-off, not the dummy env's toy one.
