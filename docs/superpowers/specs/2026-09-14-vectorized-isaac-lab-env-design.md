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

This design rewrites `IsaacLabTalonEnv` onto Isaac Lab's `DirectRLEnv` +
`gym.register` path, and propagates the resulting batch-shaped contract
through `BaseTalonEnv`, `DummyTalonEnv`, `reward.py`, `preference.py`,
`obs_stack.py`, and `training/moppo.py`.

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
2. **Env creation: `DirectRLEnv` + `gym.register`**, full standard path —
   not a hand-rolled `InteractiveScene`/`Cloner` wrapper. `IsaacLabTalonEnv`
   becomes a real Isaac Lab task (`_setup_scene`, `_apply_action`,
   `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`),
   registered as `Isaac-Talon-A1-v0`.
3. **`BaseTalonEnv` rewritten batch-native**, no adapter/wrapper layer.
   `DummyTalonEnv` also becomes batch-shaped (plain numpy, no gym) so both
   envs satisfy the same contract directly.
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
        gym VectorEnv / Isaac Lab DirectRLEnv auto-reset semantics."""
```

All transition dict values (`v_actual`, `v_command`, `obstacle_dist`,
`joint_torque`, `joint_vel`, `foot_contact_force`, `action`, `prev_action`,
`joint_acc`, `obs`) gain a leading `num_envs` axis; per-env dtype/meaning is
unchanged from the current single-env contract.

## `IsaacLabTalonEnv` → `DirectRLEnv`

- `IsaacLabTalonEnvCfg(DirectRLEnvCfg)`: `num_envs`, `env_spacing` (grid
  layout for the `N` cloned robots), the `UNITREE_A1_CFG`-derived
  `ArticulationCfg` (Kp=55/Kd=0.8 override, `base_legs` actuator group —
  both confirmed 2026-09-14), action/observation space dims from
  `ObservationSpaceCfg`/`ActionSpaceCfg`.
- `IsaacLabTalonEnv(DirectRLEnv)`:
  - `_setup_scene()`: builds the `InteractiveScene` (Isaac Lab clones the
    robot `N` times via `Cloner` internally — this is what actually gives
    `num_envs`, not code this repo writes by hand), ground plane, per-env
    `ContactSensor` on `.*_foot`.
  - `_apply_action()`: batched `set_joint_position_target` across all `N`
    envs.
  - `_get_observations()`: builds the batched `(N, obs_dim)` obs from
    `self.scene["robot"].data.*` (already batched per-env by Isaac Lab).
  - `_get_rewards()`: calls `reward.compute_reward_vector` (batched, see
    below) — returns `(N, 5)`, consistent with the rest of this repo's
    reward-vector design, not Isaac Lab's usual scalar per-env reward
    (`DirectRLEnv` tolerates a non-scalar reward return; `moppo.py` is what
    actually consumes the vector).
  - `_get_dones()`: `(N,)` bool — horizon timeout OR the same scripted
    obstacle-distance check as today, still a placeholder (no Exteroception
    Module), per env.
  - `_reset_idx(env_ids)`: Isaac Lab's per-lane reset hook — resets only the
    named lanes' robot state, called automatically by the base class for any
    lane whose `_get_dones()` was `True`.
  - `render=False` is kept for the same reason as 2026-09-13's fix — this
    prelim has no visual observation. `DirectRLEnv` respects this per its
    own `sim.step(render=...)` plumbing.
- Registration: `gym.register(id="Isaac-Talon-A1-v0", entry_point=IsaacLabTalonEnv, ...)`.
  `train_prelim.py` constructs it via `gym.make("Isaac-Talon-A1-v0", cfg=cfg)`
  — the returned object satisfies `BaseTalonEnv` directly (no wrapper).

## `DummyTalonEnv` (batch-native)

Stays physics-free (1D toy), but every array gains the leading `num_envs`
axis and per-lane auto-reset is implemented directly in numpy (no gym): each
lane tracks its own `pos`/`vel`/`obstacle_ahead`/`v_command`; on
`done[i] == True`, that lane's state is resampled/reset in-place before
`step()` returns.

## `reward.py` (batched)

Each of the 5 term functions takes batched inputs and returns `(N,)`
instead of a scalar — mechanical rewrite (`np.sum(..., axis=-1)` instead of
`np.sum(...)`, etc.), no behavior change per-lane. `compute_reward_vector`
returns `(N, 5)` instead of `(5,)`.

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
- **`_reset_idx`/auto-reset correctness is the highest-risk mechanical
  piece.** Isaac Lab's per-lane reset must not leak state (stale contact
  forces, previous episode's `prev_action`) into the fresh lane — worth an
  explicit test asserting a freshly-reset lane's transition matches a
  standalone `reset()` for that same lane.
- **This is a strictly larger, slower-to-verify change** than 2026-09-13's
  milestone — expect the manual GPU smoke test step to take longer to debug
  (auto-reset and batched GAE are both new failure surfaces that didn't
  exist in the single-env version).
