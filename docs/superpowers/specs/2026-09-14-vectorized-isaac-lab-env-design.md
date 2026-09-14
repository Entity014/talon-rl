# Vectorized Isaac Lab env — design

Date: 2026-09-14
Status: approved, pending implementation plan

## Context

`IsaacLabTalonEnv` (shipped 2026-09-14, see
`docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md`) wraps a
single `SimulationContext` + one `Articulation` directly — `num_envs=1`,
chosen deliberately for that milestone's narrow goal ("pipeline runs
headless without crashing", not a locomotion result). A single-env, 200-step
rollout with `render=False` already takes meaningful wall time; any real
training run needs Isaac Lab's standard vectorized-env pattern (`Cloner` +
batched PhysX stepping, the same mechanism `./isaaclab.sh -p
scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0
--num_envs 64 --headless` and `jaykorea/Isaac-RL-Two-wheel-Legged-Bot` use).

This design rewrites `IsaacLabTalonEnv` onto Isaac Lab's `ManagerBasedRLEnv`
+ `gym.register` path (revised mid-brainstorming from an initial
`DirectRLEnv` choice — see Decision 2), and propagates the resulting
batch-shaped contract through `BaseTalonEnv`, `DummyTalonEnv`, `reward.py`,
`preference.py`, `obs_stack.py`, and `training/moppo.py`.

**Explicitly out of scope (confirmed with user 2026-09-14):** this is
**not** a change to the thesis's RL problem formulation. The single
preference-conditioned policy (chapter3.tex's Multi-Objective Module,
`RewardVectorCfg`'s 5-term vector arbitrated by `w`) stays exactly as
designed. "Behaviors" like stand/walk/jump/run/get-up, raised during
brainstorming, are a future direction (possibly separate gym-registered
tasks later) — not part of this change. No chapter3.tex edits.

## Decisions (from brainstorming, 2026-09-14)

1. **Rollout collection: fixed-horizon + auto-reset** (Isaac
   Lab/rsl_rl/sb3-standard), not episode-based. `N` envs step in lockstep for
   a fixed `num_steps` per `update()` call; any env lane that terminates
   auto-resets internally and keeps contributing to the same rollout buffer.
2. **Env creation: `ManagerBasedRLEnv` + `gym.register`**, full standard
   path — not a hand-rolled `InteractiveScene`/`Cloner` wrapper, and
   revised from an initial `DirectRLEnv` choice once folder-tree research
   against `jaykorea/Isaac-RL-Two-wheel-Legged-Bot`
   (`lab/flamingo/tasks/manager_based/locomotion/velocity/...`) showed that
   repo — the explicit reference for "standard" here — uses
   `ManagerBasedRLEnv` throughout, not `DirectRLEnv`. `IsaacLabTalonEnv`
   becomes a real Isaac Lab manager-based task registered as
   `Isaac-Talon-A1-v0`. One deliberate deviation from the manager pattern:
   Isaac Lab's `RewardManager` always sums reward terms into one scalar per
   env, but this repo needs the unsummed 5-term **vector** (`w` arbitrates
   it downstream in `moppo.py`) — so reward terms are still written in
   manager-term shape (`func(env) -> Tensor`, living in `mdp/rewards.py`,
   matching the reference repo's file layout) but `_get_rewards()` calls
   them directly and stacks the result instead of registering them with
   `RewardTermCfg`/letting `RewardManager` sum them.
3. **`BaseTalonEnv` rewritten batch-native**, no adapter/wrapper layer for
   the real env. `DummyTalonEnv` becomes `gym.vector.SyncVectorEnv` wrapping
   a single-lane `DummyEnv(gym.Env)`, reusing gym's own tested auto-reset
   logic instead of hand-rolling it a second time — a thin adapter inside
   `DummyTalonEnv` (still `BaseTalonEnv`-shaped from the outside) converts
   `SyncVectorEnv`'s `(obs, reward, terminated, truncated, info)` tuple into
   this repo's transition dict.
4. **`num_envs`**: empirical sizing starting at **2048** directly (not a
   16→double search) against the RTX 3070 Ti's 8GB VRAM — try 4096 next if
   2048 fits with headroom, back off if either OOMs. Whatever number
   actually fits becomes the default; `--num_envs` stays overridable from
   the CLI either way.

## `BaseTalonEnv` contract (new)

```python
class BaseTalonEnv(ABC):
    num_envs: int
    obs_dim: int      # WITHOUT the preference vector — moppo.py appends w
    action_dim: int

    def reset(self) -> dict:
        """Every value in the returned dict has a leading (num_envs, ...) axis."""

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        """action: (num_envs, action_dim). Returns (transition, done), done: (num_envs,) bool.
        Any lane with done[i] == True has ALREADY been auto-reset internally —
        transition["obs"][i] is that lane's fresh post-reset observation, matching
        gym VectorEnv / Isaac Lab ManagerBasedRLEnv auto-reset semantics."""
```

All transition dict values (`v_actual`, `v_command`, `obstacle_dist`,
`joint_torque`, `joint_vel`, `foot_contact_force`, `action`, `prev_action`,
`joint_acc`, `obs`) gain a leading `num_envs` axis; per-env dtype/meaning is
unchanged from the current single-env contract.

## `IsaacLabTalonEnv` → `ManagerBasedRLEnv`

Folder layout mirrors `jaykorea/Isaac-RL-Two-wheel-Legged-Bot`'s
`lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/`
(verified against the actual repo tree 2026-09-14), adapted to this repo —
no `agents/` subfolder, since that exists in the reference repo to hold
config for several interchangeable RL libraries (rsl_rl, rl_games, co_rl)
and this repo only ever trains with its own `training/moppo.py`:

```text
talon_rl/tasks/locomotion/a1_env/
  __init__.py            # gym.register("Isaac-Talon-A1-v0", entry_point=...)
  a1_env_cfg.py           # ManagerBasedRLEnvCfg: scene + observations + actions
                           # + terminations + events cfg classes (MySceneCfg-style,
                           # per the reference repo's velocity_env_cfg.py)
  mdp/
    __init__.py
    observations.py        # term funcs: func(env: ManagerBasedRLEnv, ...) -> Tensor,
                            # consumed normally by Isaac Lab's ObservationManager
    rewards.py              # SAME signature/shape as the reference repo's
                             # mdp/rewards.py — but NOT registered via RewardTermCfg;
                             # IsaacLabTalonEnv._get_rewards() imports and calls all 5
                             # directly, np.stack()s them into (N, 5). See Decision 2.
    terminations.py         # horizon timeout + the scripted obstacle-distance check
                             # (still a placeholder — no Exteroception Module)
    events.py                # per-lane reset event (robot pose/joint state randomize
                              # on reset), Isaac Lab's manager-based reset hook
```

- `IsaacLabTalonEnvCfg(ManagerBasedRLEnvCfg)`: `MySceneCfg`-equivalent
  (`num_envs`, `env_spacing`, the `UNITREE_A1_CFG`-derived `ArticulationCfg`
  with the Kp=55/Kd=0.8 override and `base_legs` actuator group — both
  confirmed 2026-09-14 — plus the per-env `ContactSensor` on `.*_foot`),
  `ObservationsCfg`, `ActionsCfg`, `TerminationsCfg`, `EventCfg` composed
  from `mdp/*.py` term functions, matching the reference repo's
  `velocity_env_cfg.py` composition pattern.
- `IsaacLabTalonEnv(ManagerBasedRLEnv)`: the base class handles scene
  cloning (`InteractiveScene` + `Cloner`, `N` robots), the observation
  manager, the termination manager, and per-lane auto-reset (event manager)
  entirely from the cfg — this repo only overrides `_get_rewards()` to
  bypass `RewardManager`'s scalar sum and return the `(N, 5)` vector
  instead (the one deliberate deviation from the manager pattern, per
  Decision 2).
- `render=False` is kept for the same reason as 2026-09-13's fix — this
  prelim has no visual observation.
- Registration: `gym.register(id="Isaac-Talon-A1-v0", entry_point=IsaacLabTalonEnv, ...)`
  in `talon_rl/tasks/locomotion/a1_env/__init__.py`. `train_prelim.py`
  constructs it via `gym.make("Isaac-Talon-A1-v0", cfg=cfg)` — the returned
  object satisfies `BaseTalonEnv` directly (no wrapper).

## `DummyTalonEnv` (batch-native, `gym.vector.SyncVectorEnv`)

Stays physics-free (1D toy). `DummyEnv(gym.Env)` implements the single-lane
kinematics (unchanged math: `action[0]` drives forward accel, `action[1]`
softens landing) using gym's own `reset()`/`step()` contract, including
gym's own auto-reset-on-termination behavior. `DummyTalonEnv` wraps `N` of
these in `gym.vector.SyncVectorEnv` and adapts its
`(obs, reward, terminated, truncated, info)` tuple return into this
repo's transition dict — reuses gym's tested vectorization/auto-reset
instead of a second hand-rolled implementation (`IsaacLabTalonEnv`'s
auto-reset already comes for free from Isaac Lab's manager machinery, so
this keeps both real envs using a tested auto-reset implementation rather
than this repo's own).

## `reward.py` (batched) — single source of truth for reward math

Each of the 5 term functions takes batched inputs and returns `(N,)`
instead of a scalar — mechanical rewrite (`np.sum(..., axis=-1)` instead of
`np.sum(...)`, etc.), no behavior change per-lane. `compute_reward_vector`
returns `(N, 5)` instead of `(5,)`.

**No duplicate reward math in `mdp/rewards.py`.** `talon_rl/reward.py`'s
functions stay the one place the 5 reward formulas are written. The
manager-shaped functions in `talon_rl/tasks/locomotion/a1_env/mdp/rewards.py`
are thin adapters: each pulls its inputs off `env` (torch, GPU-resident),
`.cpu().numpy()`s them into the same transition-dict shape `reward.py`
already expects, and calls the matching `reward.py` function — this repo's
`BaseTalonEnv` contract already requires a CPU/numpy transition dict for
observations every step (see `IsaacLabTalonEnv(ManagerBasedRLEnv)` above),
so this adds no new GPU→CPU roundtrip beyond what already exists.

## `preference.py` (batched)

- `sample_preference_vector`: `rng.dirichlet(alpha, size=N)` → `(N, dim)`.
- `rate_limit`, `floor_clip`: vectorized over the leading `N` axis
  (broadcasting `np.linalg.norm(..., axis=-1)` etc.) — same per-lane math as
  today, applied independently per row.

## `obs_stack.py` (batched, per-lane reset)

`ObservationStack` becomes array-backed: `(N, num_stacks, obs_dim)` ring
buffer instead of a `deque`. `push(obs, done_mask)` takes the batch obs and
a `(N,)` done mask; any lane with `done_mask[i] == True` has its history
zero-filled before the new obs is pushed (mirrors today's `reset()`
zero-warm-up, but per-lane instead of whole-buffer).

## `training/moppo.py` (persistent rollout, done-masked GAE)

**Behavioral change from today:** `update()` currently is self-contained —
calls `env.reset()` and runs full episodes every call. The vectorized
version collects a **continuous** rollout: `env.reset()` happens once (at
trainer construction), and every `update()` call collects the *next*
`num_steps` timesteps across all `N` lanes, continuing wherever the
previous `update()` left off (standard on-policy vectorized-PPO pattern,
matches rsl_rl/sb3).

- `MOPPOConfig` gains `num_steps: int` (rollout length per update, e.g. 24),
  replacing `episodes_per_update`.
- Trainer holds persistent per-lane state across `update()` calls: current
  `w: (N, dim)`, the `ObservationStack`, and the last observation.
- Each collected step: any lane with `done[i] == True` (from the *previous*
  step) gets a **fresh** `w[i]` sample (new episode ⇒ new preference sample,
  per fig 3.3) instead of a rate-limited step toward the old target; lanes
  still mid-episode rate-limit toward a resampled target exactly as today.
- GAE uses the standard done-masked recursion so value bootstrapping never
  crosses an episode boundary within a lane:
  `delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)`,
  `gae_t = delta_t + gamma * lam * (1 - done_t) * gae_{t+1}`
  — computed per-objective (`K=5`) same as today's `_gae_per_objective`, just
  with a `(T, N)` done array folded in.
- The `(T, N, ...)` buffer flattens to `(T*N, ...)` before the same PPO
  epoch loop that exists today (loss/optimizer code is unchanged).

## `scripts/train_prelim.py`

Gains `--num_envs` (default = whatever the empirical sizing task lands on),
passed into `IsaacLabTalonEnvCfg`/`gym.make`. `--env dummy` keeps working
unchanged (still constructs `DummyTalonEnv` directly, now batch-native, with
the same `--num_envs` flag controlling its lane count for local testing
without a GPU).

## Testing (full rewrite, batch shapes)

Every existing test assumes the old scalar/single-env contract and needs
rewriting, not patching: `test_reward.py`, `test_preference.py`,
`test_obs_stack.py`, `test_moppo_smoke.py`, and today's
`test_isaac_lab_env.py` (which must switch from constructing
`IsaacLabTalonEnv` directly to `gym.make("Isaac-Talon-A1-v0", ...)`).
`DummyTalonEnv`'s batch behavior gets its own coverage (auto-reset
per-lane, in particular).

## Risks

- **VRAM ceiling is unverified at any of these scales.** The 2026-09-13
  design doc already flagged `num_envs=1` against the RTX 3070 Ti's 8GB as
  a known risk; 2048 (let alone 4096) has not been tried against an A1
  (17 bodies, heavier than jaykorea's two-wheel robot) on this GPU. The
  implementation plan's first task must empirically find the actual ceiling
  before anything else is built on top of an assumed `num_envs`.
- **Event-manager auto-reset correctness is the highest-risk mechanical
  piece.** `ManagerBasedRLEnv`'s reset event terms must not leak state
  (stale contact forces, previous episode's `prev_action`) into the fresh
  lane — worth an explicit test asserting a freshly-reset lane's transition
  matches a standalone `reset()` for that same lane.
- **`_get_rewards()` bypassing `RewardManager`** is the one place this repo
  departs from the manager pattern the rest of the env follows — worth a
  code comment at that exact override explaining why (vector vs. Isaac
  Lab's scalar-sum convention), so it doesn't read as an oversight later.
- **This is a strictly larger, slower-to-verify change** than 2026-09-13's
  milestone — expect the manual GPU smoke test step to take longer to debug
  (auto-reset and batched GAE are both new failure surfaces that didn't
  exist in the single-env version).
