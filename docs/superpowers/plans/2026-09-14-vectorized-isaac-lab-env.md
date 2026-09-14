# Vectorized Isaac Lab Env Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `IsaacLabTalonEnv` onto Isaac Lab's `ManagerBasedRLEnv` +
`gym.register` (mirroring `jaykorea/Isaac-RL-Two-wheel-Legged-Bot`'s
`tasks/manager_based/locomotion/velocity/<robot>_env/` + `mdp/` folder
convention), and propagate the resulting batch-native `(num_envs, ...)`
contract through `BaseTalonEnv`, `DummyTalonEnv`, `reward.py`,
`preference.py`, `obs_stack.py`, and `training/moppo.py` — replacing
2026-09-13's single-env (`num_envs=1`) smoke-test env with a real vectorized
one, so an actual training run doesn't take single-env wall time.

**Architecture:** `IsaacLabTalonEnv(ManagerBasedRLEnv, BaseTalonEnv)` gets
scene cloning, the observation manager, the termination manager, and
per-lane auto-reset entirely from `IsaacLabTalonEnvCfg` (standard Isaac Lab
manager-based task) — the only overrides are `step()`/`reset()`, which
translate Isaac Lab's gym-native return shape into this repo's
`(transition_dict, done_array)` contract and compute the unsummed 5-term
reward **vector** via `talon_rl.reward.compute_reward_vector()` directly
(bypassing `RewardManager`'s scalar-sum contract, which can't produce a
vector — `RewardsCfg` stays deliberately empty). `DummyTalonEnv` wraps
`gym.vector.SyncVectorEnv` around a single-lane `DummyEnv(gym.Env)`, reusing
gym's own auto-reset instead of hand-rolling it. `training/moppo.py` moves
from episode-based rollout collection to a persistent fixed-horizon +
auto-reset loop with done-masked GAE (the rsl_rl/sb3-standard pattern).

**Tech Stack:** Isaac Sim 5.0.0.0, Isaac Lab `release/2.3.0` (isaaclab
0.48.0), Python 3.11, `~/isaac-lab-env` venv, PyTorch, `gymnasium`.

**Spec:** `docs/superpowers/specs/2026-09-14-vectorized-isaac-lab-env-design.md`

## Global Constraints

- `num_envs` default: whatever Task 1's empirical VRAM test finds fits on
  the RTX 3070 Ti's 8GB, starting the search at 2048 — no task after Task 1
  may hardcode a `num_envs` default without that number.
- `BaseTalonEnv.obs_dim` **excludes** the preference vector `w` — never
  append it inside any env; `training/moppo.py` does this itself.
- Actuator group key is `"base_legs"`, PD gains `Kp=55.0`/`Kd=0.8` — both
  confirmed against the real installed `UNITREE_A1_CFG` (2026-09-13). Do not
  reintroduce the wrong `"legs"` guess.
- No duplicate reward math: `talon_rl/reward.py`'s 5 functions are the one
  place the reward formulas are written. Nothing under
  `talon_rl/tasks/locomotion/a1_env/` may reimplement them — call
  `talon_rl.reward`'s functions instead.
- No locomotion-result claims anywhere (README's existing rule, and the
  2026-09-13 design doc's) — "doesn't crash, obs/reward are finite" is the
  bar every manual GPU test in this plan targets, not a training result.
- `render=False` behavior: do not add a manual `sim.step(render=False)`
  override — confirmed (2026-09-14, reading
  `isaaclab/envs/manager_based_env.py:427-459`) that `ManagerBasedEnv.step()`
  already skips rendering automatically whenever there's no GUI and no RTX
  sensors, which is always true for this headless, camera-free env.

---

### Task 1: Empirical `num_envs` VRAM sizing

**Files:** none in this repo — throwaway script against the host machine.

**Interfaces:**
- Produces: a verified `num_envs` value (int) that Task 6 hardcodes as
  `IsaacLabTalonEnvCfg`'s default and Task 8 hardcodes as
  `scripts/train_prelim.py --num_envs`'s default.

- [ ] **Step 1: Write and run the sizing script**

```python
# /tmp/vram_sizing.py
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import sys
num_envs = int(sys.argv[1])

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import carb
from isaacsim.storage.native import get_assets_root_path
_root = get_assets_root_path()
if _root is None:
    raise RuntimeError("Could not resolve Isaac Sim assets root")
carb.settings.get_settings().set("/persistent/isaac/asset_root/cloud", _root)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import UNITREE_A1_CFG

@configclass
class SizingSceneCfg(InteractiveSceneCfg):
    robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02))
sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
scene = InteractiveScene(SizingSceneCfg(num_envs=num_envs, env_spacing=2.5))
sim.reset()
for _ in range(5):
    sim.step(render=False)

import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
with open("/tmp/vram_sizing_result.txt", "a") as f:
    f.write(f"num_envs={num_envs} allocated={allocated:.2f}GB reserved={reserved:.2f}GB\n")

# SimulationApp.close() is known to hang after a GPU-pipeline scene has been
# stepped (2026-09-14) — this script's job is done once the result line is
# written, so skip close() and just end the process.
os._exit(0)
```

Run:
```bash
source ~/isaac-lab-env/bin/activate
rm -f /tmp/vram_sizing_result.txt
python /tmp/vram_sizing.py 2048
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
cat /tmp/vram_sizing_result.txt
```
Expected: script exits (any exit code — `os._exit(0)` at the end always
returns 0, so use the presence of a `num_envs=2048 ...` line in the result
file as the actual success signal, not the process exit code alone) and
prints a `num_envs=2048 allocated=...GB reserved=...GB` line;
`nvidia-smi`'s `memory.used` should be comfortably under 8192 MiB (leave at
least ~1GB headroom for the training process's own PyTorch model + rollout
buffers, which Task 6/8 add on top of this baseline scene cost).

- [ ] **Step 2: If Step 1 fits with headroom, try 4096**

Run: `python /tmp/vram_sizing.py 4096` (same venv, same nvidia-smi check).
If this also fits with headroom, 4096 is the sizing result. If it OOMs
(`RuntimeError: CUDA out of memory` in the script's stdout, or the process
is killed) or leaves under ~1GB free, 2048 is the result instead.

- [ ] **Step 3: If 2048 itself doesn't fit, back off**

Run: `python /tmp/vram_sizing.py 1024`, then `512` if needed, halving until
one fits with headroom. Record whichever value actually worked.

- [ ] **Step 4: Record the result**

Note the final chosen `num_envs` value — Task 6 Step 3 and Task 8 Step 1
both hardcode it as their default. No commit for this task (the script is
throwaway, not part of the repo); the result feeds directly into later
tasks' code.

---

### Task 2: `BaseTalonEnv` batch contract + `DummyTalonEnv` rewrite

**Files:**
- Modify: `talon_rl/envs/base_env.py` (full rewrite)
- Modify: `talon_rl/envs/dummy_env.py` (full rewrite)
- Test: `tests/test_dummy_env.py` (new)

**Interfaces:**
- Produces: `BaseTalonEnv` contract — `num_envs: int`, `obs_dim: int`,
  `action_dim: int`, `reset() -> dict` (every value `(num_envs, ...)`),
  `step(action: (num_envs, action_dim)) -> tuple[dict, done: (num_envs,) bool]`.
  `DummyTalonEnv(obs_cfg, action_cfg, num_envs=1, horizon=200, dt=0.02, seed=0)`
  satisfies it directly. Task 6 (`IsaacLabTalonEnv`) and Task 7
  (`training/moppo.py`) both consume this contract.

- [ ] **Step 1: Rewrite `base_env.py`**

```python
# talon_rl/envs/base_env.py
"""Env interface contract — batch-native (num_envs, ...) shapes throughout.

Both DummyTalonEnv and IsaacLabTalonEnv (constructed via
gym.make("Isaac-Talon-A1-v0")) satisfy this directly, no wrapper.
reset()/step() return every transition value with a leading
(num_envs, ...) axis; step()'s `done` array marks which lanes just
auto-reset internally — that lane's "obs" row (and every other field) is
already the fresh post-reset value, not the terminal one, matching gym
VectorEnv / Isaac Lab ManagerBasedRLEnv auto-reset semantics.

A transition dict must carry every key talon_rl.reward._TERM_FUNCS expects
(v_actual, v_command, obstacle_dist, joint_torque, joint_vel,
foot_contact_force, action, prev_action, joint_acc), plus "obs" — see
reward.py for the exact shapes each key needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTalonEnv(ABC):
    num_envs: int
    obs_dim: int  # WITHOUT the preference vector — moppo.py appends w itself
    action_dim: int

    @abstractmethod
    def reset(self) -> dict:
        """Returns the first transition dict; every value has a leading (num_envs, ...) axis."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        """action: (num_envs, action_dim). Returns (transition, done); done: (num_envs,) bool.
        Any lane with done[i] == True has already been auto-reset internally."""
```

- [ ] **Step 2: Write the failing tests for `DummyTalonEnv`**

```python
# tests/test_dummy_env.py
import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.envs.dummy_env import DummyTalonEnv


def _make_env(num_envs=4, horizon=5, seed=0):
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=num_envs, horizon=horizon, seed=seed)
    return env, obs_cfg, action_cfg


def test_satisfies_base_contract_and_shapes():
    env, obs_cfg, action_cfg = _make_env(num_envs=4)
    assert isinstance(env, BaseTalonEnv)
    assert env.num_envs == 4
    assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
    assert env.action_dim == action_cfg.dim

    transition = env.reset()
    assert transition["obs"].shape == (4, env.obs_dim)
    for key in (
        "v_actual", "v_command", "obstacle_dist", "joint_torque", "joint_vel",
        "foot_contact_force", "action", "prev_action", "joint_acc",
    ):
        assert key in transition
        assert transition[key].shape[0] == 4

    action = np.zeros((4, env.action_dim), dtype=np.float32)
    transition, done = env.step(action)
    assert transition["obs"].shape == (4, env.obs_dim)
    assert done.shape == (4,)
    assert done.dtype == bool


def test_lanes_auto_reset_independently():
    """horizon=1 forces every lane to terminate every single step — after
    enough steps, each lane must have resampled its own v_command/obstacle
    (proves auto-reset runs per-lane, not just once globally)."""
    env, _, action_cfg = _make_env(num_envs=8, horizon=1, seed=0)
    env.reset()
    action = np.zeros((8, action_cfg.dim), dtype=np.float32)

    v_commands_seen = set()
    for _ in range(10):
        transition, done = env.step(action)
        assert np.all(done)  # horizon=1: every lane terminates every step
        for row in transition["v_command"]:
            v_commands_seen.add(tuple(np.round(row, 4)))

    # 8 lanes x 10 auto-resets each, sampled from a continuous range -> expect
    # meaningfully more than 1 distinct v_command, proving lanes actually
    # resample independently rather than sharing one global RNG draw.
    assert len(v_commands_seen) > 5


def test_smoothness_uses_correct_prev_action_ordering():
    """joint_acc = (action - prev_action) / dt must use the PREVIOUS step's
    action, not this step's — regression test for the prev_action snapshot
    timing bug caught while writing this env (see DummyEnv._compute_transition)."""
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    env.reset()

    a1 = np.full((1, action_cfg.dim), 0.5, dtype=np.float32)
    t1, _ = env.step(a1)
    assert np.allclose(t1["prev_action"], 0.0)  # first step: prev was the reset zero-action
    assert np.allclose(t1["action"], 0.5)

    a2 = np.full((1, action_cfg.dim), -0.3, dtype=np.float32)
    t2, _ = env.step(a2)
    assert np.allclose(t2["prev_action"], 0.5)  # second step: prev is step 1's action
    assert np.allclose(t2["action"], -0.3)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dummy_env.py -v`
Expected: FAIL (`ImportError`/`AttributeError` — `DummyTalonEnv` doesn't
accept `num_envs` yet).

- [ ] **Step 4: Rewrite `dummy_env.py`**

```python
# talon_rl/envs/dummy_env.py
"""Physics-free smoke-test env — batch-native, gym.vector.SyncVectorEnv-backed.

Still NOT a locomotion simulator (see the original 2026-09-11 docstring for
the "why" of the toy physics) — the only change here is vectorization.
DummyEnv(gym.Env) implements one lane's kinematics using gym's own
reset()/step() contract, and DummyTalonEnv wraps `num_envs` of them in
gym.vector.SyncVectorEnv — this reuses gym's own tested per-lane auto-reset
logic instead of hand-rolling it a second time (IsaacLabTalonEnv's
auto-reset already comes from Isaac Lab's manager machinery for free, so
both real envs now lean on a tested implementation rather than this repo's
own). gymnasium's automatic info-dict vectorization doesn't cleanly carry
array-valued fields (only scalar diagnostics), so DummyTalonEnv reads the
extra reward-computation fields directly off each sub-env's post-step
instance state (`self._vec_env.envs`) instead of through `info`.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import ActionSpaceCfg, ObservationSpaceCfg
from .base_env import BaseTalonEnv

_TRANSITION_KEYS = (
    "v_actual", "v_command", "obstacle_dist", "joint_torque", "joint_vel",
    "joint_acc", "foot_contact_force", "action", "prev_action_out",
)


class DummyEnv(gym.Env):
    """Single lane. action[0] drives forward accel, action[1] softens
    landing (reduces impact, costs extra energy) — see 2026-09-11's
    DummyTalonEnv docstring for the full rationale, unchanged here."""

    def __init__(self, obs_cfg: ObservationSpaceCfg, action_cfg: ActionSpaceCfg, horizon: int = 200, dt: float = 0.02):
        super().__init__()
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.horizon = horizon
        self.dt = dt
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0.0
        self.vel = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.v_command = np.array([self.np_random.uniform(0.3, 1.0), 0.0, 0.0], dtype=np.float32)
        self.obstacle_ahead = float(self.np_random.uniform(3.0, 6.0))
        obs = self._compute_transition(np.zeros(self.action_dim, dtype=np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        accel = float(action[0]) * 2.0
        self.vel = float(np.clip(self.vel + accel * self.dt, 0.0, 2.0))
        self.pos += self.vel * self.dt
        self.t += 1

        obs = self._compute_transition(action)
        self.prev_action = action  # advance AFTER _compute_transition reads the old value

        terminated = bool(self.pos >= self.obstacle_ahead)
        truncated = bool(self.t >= self.horizon)
        return obs, 0.0, terminated, truncated, {}

    def _compute_transition(self, action: np.ndarray) -> np.ndarray:
        """Computes and stores this step's transition fields (read externally
        by DummyTalonEnv after step()/reset() return) using self.prev_action's
        CURRENT (pre-advance) value — must run before step() advances
        self.prev_action to `action`, or prev_action_out would equal this
        step's action instead of the previous one."""
        softness = float(np.clip(action[1], 0.0, 1.0)) if action.size > 1 else 0.0

        self.v_actual = np.array([self.vel, 0.0, 0.0], dtype=np.float32)
        self.v_command_out = self.v_command
        self.obstacle_dist = max(0.0, self.obstacle_ahead - self.pos)
        self.joint_vel = np.full(self.action_dim, self.vel, dtype=np.float32)
        self.joint_torque = action * (1.0 + softness)
        self.joint_acc = (action - self.prev_action) / self.dt
        self.foot_contact_force = np.full(4, self.vel * 40.0 * (1.0 - 0.5 * softness), dtype=np.float32)
        self.action = action
        self.prev_action_out = self.prev_action

        obs = np.concatenate([
            np.zeros(self.obs_cfg.joint_pos_dim, dtype=np.float32),
            np.full(self.obs_cfg.joint_vel_dim, self.vel, dtype=np.float32),
            np.zeros(self.obs_cfg.roll_pitch_dim, dtype=np.float32),
            np.ones(self.obs_cfg.foot_contact_dim, dtype=np.float32),
            self.prev_action[: self.obs_cfg.prev_action_dim],
            self.v_command,
        ])
        return obs.astype(np.float32)


class DummyTalonEnv(BaseTalonEnv):
    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        num_envs: int = 1,
        horizon: int = 200,
        dt: float = 0.02,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self._seed = seed
        self._vec_env = gym.vector.SyncVectorEnv(
            [(lambda: DummyEnv(obs_cfg, action_cfg, horizon=horizon, dt=dt)) for _ in range(num_envs)]
        )

    def reset(self) -> dict:
        obs, _infos = self._vec_env.reset(seed=[self._seed + i for i in range(self.num_envs)])
        return self._collect_transition(obs)

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        obs, _rewards, terminated, truncated, _infos = self._vec_env.step(action)
        done = np.logical_or(terminated, truncated)
        return self._collect_transition(obs), done

    def _collect_transition(self, obs: np.ndarray) -> dict:
        transition = {"obs": obs.astype(np.float32)}
        for key in ("v_actual", "obstacle_dist", "joint_torque", "joint_vel",
                    "joint_acc", "foot_contact_force", "action"):
            transition[key] = np.stack([getattr(env.unwrapped, key) for env in self._vec_env.envs])
        transition["v_command"] = np.stack([env.unwrapped.v_command_out for env in self._vec_env.envs])
        transition["prev_action"] = np.stack([env.unwrapped.prev_action_out for env in self._vec_env.envs])
        return transition

    def close(self) -> None:
        self._vec_env.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dummy_env.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Update the old single-env dummy-env test file**

`tests/test_moppo_smoke.py` still imports `DummyTalonEnv(obs_cfg, action_cfg, horizon=40, seed=0)` without `num_envs` — it will be rewritten in Task 7 alongside `moppo.py` (it exercises both together). Leave it failing for now; do not fix it here.

- [ ] **Step 7: Commit**

```bash
git add talon_rl/envs/base_env.py talon_rl/envs/dummy_env.py tests/test_dummy_env.py
git commit -m "feat: batch-native BaseTalonEnv contract, DummyTalonEnv via gym.vector.SyncVectorEnv"
```

---

### Task 3: `reward.py` batched

**Files:**
- Modify: `talon_rl/reward.py` (full rewrite)
- Modify: `tests/test_reward.py` (full rewrite)

**Interfaces:**
- Consumes: nothing new.
- Produces: `progress_reward`, `clearance_reward`, `energy_reward`,
  `impact_reward`, `smoothness_reward` — each now takes batched
  `(N, ...)` inputs and returns `(N,)`. `compute_reward_vector(transition, cfg) -> (N, 5)`
  (was `(5,)`). Consumed by Task 7 (`moppo.py`) and Task 6
  (`IsaacLabTalonEnv.step()`), both via `compute_reward_vector`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reward.py
import numpy as np

from talon_rl.config import RewardVectorCfg
from talon_rl.reward import (
    clearance_reward,
    compute_reward_vector,
    energy_reward,
    impact_reward,
    progress_reward,
    smoothness_reward,
)


def test_progress_reward_is_max_at_zero_error():
    v = np.tile(np.array([0.5, 0.0, 0.0]), (3, 1))
    r_same = progress_reward(v, v, std=0.5)
    r_diff = progress_reward(v, v * 2, std=0.5)
    assert r_same.shape == (3,)
    assert np.allclose(r_same, 1.0)
    assert np.all(r_diff < 1.0)


def test_clearance_reward_clips_to_unit_interval():
    dist = np.array([10.0, 0.0])
    r = clearance_reward(dist, safe_dist=0.5)
    assert r.shape == (2,)
    assert r[0] == 1.0
    assert r[1] == 0.0


def test_energy_reward_is_nonpositive():
    torque = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    vel = np.ones((2, 3))
    r = energy_reward(torque, vel)
    assert r.shape == (2,)
    assert r[0] <= 0.0
    assert r[1] == 0.0


def test_impact_reward_penalizes_only_above_threshold():
    forces = np.array([[10.0, 10.0, 10.0, 10.0], [200.0, 0.0, 0.0, 0.0]])
    r = impact_reward(forces, threshold=50.0)
    assert r.shape == (2,)
    assert r[0] == 0.0
    assert r[1] < 0.0


def test_smoothness_reward_penalizes_action_change():
    a = np.zeros((2, 12))
    b = np.ones((2, 12))
    acc = np.zeros((2, 12))
    r_same = smoothness_reward(a, a, acc)
    r_diff = smoothness_reward(b, a, acc)
    assert r_same.shape == (2,)
    assert np.allclose(r_same, 0.0)
    assert np.all(r_diff < 0.0)


def test_compute_reward_vector_respects_active_mask_and_order():
    cfg = RewardVectorCfg(active=(True, False, True, False, True))
    n = 4
    transition = {
        "obs": np.zeros((n, 1)),  # only used by compute_reward_vector to infer N
        "v_actual": np.zeros((n, 3)), "v_command": np.zeros((n, 3)),
        "obstacle_dist": np.ones(n),
        "joint_torque": np.ones((n, 12)), "joint_vel": np.ones((n, 12)),
        "foot_contact_force": np.zeros((n, 4)),
        "action": np.zeros((n, 12)), "prev_action": np.zeros((n, 12)), "joint_acc": np.zeros((n, 12)),
    }
    r = compute_reward_vector(transition, cfg)
    assert r.shape == (n, 5)
    assert np.allclose(r[:, 1], 0.0)  # clearance masked off
    assert np.allclose(r[:, 3], 0.0)  # impact masked off
    assert np.all(r[:, 0] != 0.0)     # progress active
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_reward.py -v`
Expected: FAIL (shape assertion errors — current functions return scalars).

- [ ] **Step 3: Rewrite `reward.py`**

```python
# talon_rl/reward.py
"""The 5-term reward vector from chapter3.tex table 3.3 (tab:reward-vector).
Batched (N, ...) -> (N,) throughout — this is the one place the 5 reward
formulas are written (talon_rl/tasks/locomotion/a1_env/'s manager-based env
calls compute_reward_vector directly rather than reimplementing any of this
in torch — see that module's docstring).

Each function takes batched arrays describing the current transition and
returns a per-lane (N,) float32 array; `compute_reward_vector` stacks them
into a fixed-order (N, dim) array matching RewardVectorCfg.term_names.

None of these terms know about the preference vector w — arbitration between
them happens downstream (see training/moppo.py), consistent with chapter3.tex
treating w as something applied to the reward *vector*, not baked into any
single term.
"""

from __future__ import annotations

import numpy as np

from .config import RewardVectorCfg


def progress_reward(v_actual: np.ndarray, v_command: np.ndarray, std: float) -> np.ndarray:
    """Go-anywhere navigation: exp-kernel velocity tracking. (N, 3), (N, 3) -> (N,)."""
    err = np.sum((v_actual - v_command) ** 2, axis=-1)
    return np.exp(-err / (std**2)).astype(np.float32)


def clearance_reward(obstacle_dist: np.ndarray, safe_dist: float = 0.5) -> np.ndarray:
    """Autonomous obstacle negotiation — scripted signal until the Exteroception
    Module exists (out of scope here). (N,) -> (N,)."""
    return np.clip(obstacle_dist / safe_dist, 0.0, 1.0).astype(np.float32)


def energy_reward(joint_torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Raw power per step, negated. (N, 12), (N, 12) -> (N,)."""
    power = np.sum(np.abs(joint_torque * joint_vel), axis=-1)
    return (-power).astype(np.float32)


def impact_reward(foot_contact_force: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Continuous impact mitigation — penalize peak landing force, not just
    falls. Threshold is an arbitrary prelim placeholder; retune against real
    A1 contact-force ranges before any hardware test. (N, 4) -> (N,)."""
    if foot_contact_force.shape[-1] == 0:
        return np.zeros(foot_contact_force.shape[0], dtype=np.float32)
    peak = np.max(np.abs(foot_contact_force), axis=-1)
    return (-np.maximum(0.0, peak - threshold) / threshold).astype(np.float32)


def smoothness_reward(action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray) -> np.ndarray:
    """Fixed-weight regularizer (table 3.3's 5th row) — NOT part of the
    preference vector w in the real system, but kept in the vector here so
    the smoke test exercises all 5 terms end-to-end. (N, 12) each -> (N,)."""
    action_rate = np.sum((action - prev_action) ** 2, axis=-1)
    acc = np.sum(joint_acc**2, axis=-1)
    return (-(action_rate + 0.01 * acc)).astype(np.float32)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(t["v_actual"], t["v_command"], cfg.progress_std),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(t["foot_contact_force"]),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"]),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_reward.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add talon_rl/reward.py tests/test_reward.py
git commit -m "feat: batch reward.py to (N, ...) -> (N,) throughout"
```

---

### Task 4: `preference.py` batched

**Files:**
- Modify: `talon_rl/preference.py` (full rewrite)
- Modify: `tests/test_preference.py` (full rewrite)

**Interfaces:**
- Produces: `sample_preference_vector(rng, reward_cfg, pref_cfg, num_envs) -> (N, dim)`,
  `rate_limit(w_prev: (N,dim), w_target: (N,dim), max_delta) -> (N,dim)`,
  `floor_clip(w: (N,dim), term_names, floor_eps, floored_term="impact") -> (N,dim)`.
  Consumed by Task 7 (`moppo.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preference.py
import numpy as np

from talon_rl.config import PreferenceCfg, RewardVectorCfg
from talon_rl.preference import floor_clip, rate_limit, sample_preference_vector


def test_sample_preference_vector_sums_to_one_and_matches_shape():
    rng = np.random.default_rng(0)
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    w = sample_preference_vector(rng, reward_cfg, pref_cfg, num_envs=5)
    assert w.shape == (5, reward_cfg.dim)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(w >= 0.0)


def test_rate_limit_caps_step_size_per_row():
    w_prev = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (3, 1))
    w_target = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (3, 1))
    w_next = rate_limit(w_prev, w_target, max_delta=0.05)
    norms = np.linalg.norm(w_next - w_prev, axis=-1)
    assert np.all(norms <= 0.05 + 1e-6)


def test_rate_limit_passes_through_rows_within_cap():
    w_prev = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (2, 1))
    w_target = np.array([[0.21, 0.2, 0.2, 0.2, 0.19], [0.2, 0.2, 0.2, 0.2, 0.2]])
    w_next = rate_limit(w_prev, w_target, max_delta=1.0)
    assert np.allclose(w_next, w_target)


def test_rate_limit_handles_mixed_rows_independently():
    """One row needs capping, the other doesn't — each row's cap decision
    must not affect the other (the original bug this guards: a naive
    scalar-norm implementation applied uniformly across the whole batch)."""
    w_prev = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (2, 1))
    w_target = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.21, 0.2, 0.2, 0.2, 0.19]])
    w_next = rate_limit(w_prev, w_target, max_delta=0.05)
    assert np.linalg.norm(w_next[0] - w_prev[0]) <= 0.05 + 1e-6
    assert np.allclose(w_next[1], w_target[1])  # row 1 was within cap, passes through


def test_floor_clip_enforces_minimum_and_renormalizes():
    term_names = ("progress", "clearance", "energy", "impact", "smoothness")
    w = np.array([[0.5, 0.3, 0.2, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    w_clipped = floor_clip(w, term_names, floor_eps=0.05, floored_term="impact")
    idx = term_names.index("impact")
    assert np.all(w_clipped[:, idx] >= 0.05 - 1e-6)
    assert np.allclose(w_clipped.sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(w_clipped[1], w[1], atol=1e-5)  # row already above floor: no-op
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_preference.py -v`
Expected: FAIL (`TypeError`/shape errors — current functions are single-row).

- [ ] **Step 3: Rewrite `preference.py`**

```python
# talon_rl/preference.py
"""Preference vector w: sampling, rate-limiting, floor-clip — batched (N, dim).
Mirrors the Multi-Objective Module pipeline in chapter3.tex fig 3.3 — Dirichlet
sample -> rate-limiter -> floor-clip -> (OOD monitor, not implemented here since
it depends on sigma_t from the Adaptation Module, which is out of prelim scope).
"""

from __future__ import annotations

import numpy as np

from .config import PreferenceCfg, RewardVectorCfg


def sample_preference_vector(
    rng: np.random.Generator, reward_cfg: RewardVectorCfg, pref_cfg: PreferenceCfg, num_envs: int
) -> np.ndarray:
    """One Dirichlet(alpha) sample per lane, per chapter3.tex §3.2.3 ("w ~ Dirichlet(1.0)")."""
    alpha = np.full(reward_cfg.dim, pref_cfg.dirichlet_alpha, dtype=np.float64)
    return rng.dirichlet(alpha, size=num_envs).astype(np.float32)


def rate_limit(w_prev: np.ndarray, w_target: np.ndarray, max_delta: float) -> np.ndarray:
    """Caps ||w_t - w_{t-1}|| per row so w can't jump discontinuously within
    an episode. w_prev, w_target: (N, dim) -> (N, dim)."""
    delta = w_target - w_prev
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    within_cap = (norm <= max_delta) | (norm == 0.0)
    scale = max_delta / np.maximum(norm, 1e-12)
    capped = w_prev + delta * scale
    return np.where(within_cap, w_target, capped).astype(np.float32)


def floor_clip(
    w: np.ndarray, term_names: tuple[str, ...], floor_eps: float, floored_term: str = "impact"
) -> np.ndarray:
    """w_impact >= eps always (chapter3.tex: "การตัดค่าต่ำสุด, w_impact >= epsilon") —
    never let impact-mitigation weight hit exactly zero, then renormalize each
    row to sum to 1. w: (N, dim) -> (N, dim)."""
    w = w.copy()
    idx = term_names.index(floored_term)
    below = w[:, idx] < floor_eps
    deficit = np.where(below, floor_eps - w[:, idx], 0.0)
    w[:, idx] = np.where(below, floor_eps, w[:, idx])

    others = np.array([i for i in range(w.shape[1]) if i != idx])
    other_sum = w[:, others].sum(axis=-1, keepdims=True)
    has_room = other_sum > 0
    safe_other_sum = np.where(has_room, other_sum, 1.0)
    reduction = deficit[:, None] * (w[:, others] / safe_other_sum) * has_room
    w[:, others] -= reduction

    return (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_preference.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add talon_rl/preference.py tests/test_preference.py
git commit -m "feat: batch preference.py to (N, dim) throughout"
```

---

### Task 5: `obs_stack.py` batched

**Files:**
- Modify: `talon_rl/obs_stack.py` (full rewrite)
- Modify: `tests/test_obs_stack.py` (full rewrite)

**Interfaces:**
- Produces: `ObservationStack(num_envs, obs_dim, num_policy_stacks=1, num_critic_stacks=1)`,
  `.reset(obs: (N, obs_dim))`, `.push(obs: (N, obs_dim), done_mask: (N,) bool)`,
  `.policy_obs -> (N, policy_obs_dim)`, `.critic_obs -> (N, critic_obs_dim)`.
  Consumed by Task 7 (`moppo.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_obs_stack.py
import numpy as np
import pytest

from talon_rl.obs_stack import ObservationStack


def test_default_stack_of_one_is_passthrough():
    stack = ObservationStack(num_envs=2, obs_dim=4, num_policy_stacks=1, num_critic_stacks=1)
    obs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    stack.reset(obs)
    assert np.allclose(stack.policy_obs, obs)
    assert np.allclose(stack.critic_obs, obs)


def test_reset_zero_pads_history_then_pushes_current_frame():
    stack = ObservationStack(num_envs=1, obs_dim=2, num_policy_stacks=3, num_critic_stacks=1)
    obs = np.array([[5.0, 6.0]], dtype=np.float32)
    stack.reset(obs)
    assert np.allclose(stack.policy_obs, [[0, 0, 0, 0, 5, 6]])
    assert np.allclose(stack.critic_obs, obs)


def test_push_slides_the_window_per_lane():
    stack = ObservationStack(num_envs=2, obs_dim=1, num_policy_stacks=2, num_critic_stacks=2)
    stack.reset(np.array([[1.0], [10.0]], dtype=np.float32))
    stack.push(np.array([[2.0], [20.0]], dtype=np.float32), done_mask=np.zeros(2, dtype=bool))
    assert np.allclose(stack.policy_obs, [[1.0, 2.0], [10.0, 20.0]])
    stack.push(np.array([[3.0], [30.0]], dtype=np.float32), done_mask=np.zeros(2, dtype=bool))
    assert np.allclose(stack.policy_obs, [[2.0, 3.0], [20.0, 30.0]])


def test_push_with_done_mask_resets_only_that_lane():
    stack = ObservationStack(num_envs=2, obs_dim=1, num_policy_stacks=2, num_critic_stacks=2)
    stack.reset(np.array([[1.0], [10.0]], dtype=np.float32))
    stack.push(np.array([[2.0], [20.0]], dtype=np.float32), done_mask=np.zeros(2, dtype=bool))
    # lane 0 terminates and auto-resets with a fresh obs; lane 1 keeps sliding
    stack.push(np.array([[99.0], [30.0]], dtype=np.float32), done_mask=np.array([True, False]))
    assert np.allclose(stack.policy_obs[0], [0.0, 99.0])   # lane 0: history wiped, fresh frame
    assert np.allclose(stack.policy_obs[1], [20.0, 30.0])  # lane 1: unaffected, kept sliding


def test_policy_and_critic_stacks_can_differ():
    stack = ObservationStack(num_envs=3, obs_dim=1, num_policy_stacks=4, num_critic_stacks=1)
    assert stack.policy_obs_dim == 4
    assert stack.critic_obs_dim == 1


def test_invalid_stack_count_rejected():
    with pytest.raises(ValueError):
        ObservationStack(num_envs=1, obs_dim=1, num_policy_stacks=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_obs_stack.py -v`
Expected: FAIL (`TypeError` — current constructor doesn't take `num_envs`).

- [ ] **Step 3: Rewrite `obs_stack.py`**

```python
# talon_rl/obs_stack.py
"""num_policy_stacks / num_critic_stacks pattern, after Flamingo
(jaykorea/Isaac-RL-Two-wheel-Legged-Bot) — batched (N, stacks, obs_dim).

Keeps separate temporal-history buffers for the actor and the critic, since
the two don't need the same context length: RMA's Adaptation Module needs
~0.5s of proprioceptive history \\cite{kumar2021} to regress $\\hat z_t$, but
a vector critic evaluating $V(s,c,w)$ doesn't have that requirement and can
run on fewer (or just the current) frame. Splitting this out as its own
wrapper — rather than baking stacking into the env — means `BaseTalonEnv`
implementations stay single-timestep and simple; only `training/moppo.py`
needs to know stacking exists.

`push(obs, done_mask)` zero-fills any lane whose `done_mask[i]` is True
before pushing its new frame — a fresh episode shouldn't see the previous
episode's stale history (mirrors `reset()`'s zero-warm-up, but per-lane).
"""

from __future__ import annotations

import numpy as np


class ObservationStack:
    def __init__(self, num_envs: int, obs_dim: int, num_policy_stacks: int = 1, num_critic_stacks: int = 1):
        if num_policy_stacks < 1 or num_critic_stacks < 1:
            raise ValueError("stack counts must be >= 1 (1 = no stacking, current frame only)")
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_policy_stacks = num_policy_stacks
        self.num_critic_stacks = num_critic_stacks
        self._policy_buf = np.zeros((num_envs, num_policy_stacks, obs_dim), dtype=np.float32)
        self._critic_buf = np.zeros((num_envs, num_critic_stacks, obs_dim), dtype=np.float32)

    @property
    def policy_obs_dim(self) -> int:
        return self.obs_dim * self.num_policy_stacks

    @property
    def critic_obs_dim(self) -> int:
        return self.obs_dim * self.num_critic_stacks

    def reset(self, obs: np.ndarray) -> None:
        """obs: (num_envs, obs_dim) — resets every lane's history to zero,
        then pushes the first real observation into all of them."""
        self._policy_buf[:] = 0.0
        self._critic_buf[:] = 0.0
        self.push(obs, done_mask=np.ones(self.num_envs, dtype=bool))

    def push(self, obs: np.ndarray, done_mask: np.ndarray) -> None:
        """obs: (num_envs, obs_dim), done_mask: (num_envs,) bool."""
        self._policy_buf[done_mask] = 0.0
        self._critic_buf[done_mask] = 0.0
        self._policy_buf = np.roll(self._policy_buf, shift=-1, axis=1)
        self._critic_buf = np.roll(self._critic_buf, shift=-1, axis=1)
        self._policy_buf[:, -1, :] = obs
        self._critic_buf[:, -1, :] = obs

    @property
    def policy_obs(self) -> np.ndarray:
        return self._policy_buf.reshape(self.num_envs, -1)

    @property
    def critic_obs(self) -> np.ndarray:
        return self._critic_buf.reshape(self.num_envs, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_obs_stack.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add talon_rl/obs_stack.py tests/test_obs_stack.py
git commit -m "feat: batch obs_stack.py to (N, stacks, obs_dim), per-lane reset"
```

---

### Task 6: `talon_rl/tasks/locomotion/a1_env/` — `IsaacLabTalonEnv(ManagerBasedRLEnv)`

**Files:**
- Delete: `talon_rl/envs/isaac_lab_env.py` (the 2026-09-13 single-env version)
- Delete: `tests/test_isaac_lab_env.py` (rewritten below at a new path)
- Create: `talon_rl/assets/__init__.py`
- Create: `talon_rl/assets/a1.py`
- Create: `talon_rl/tasks/__init__.py`
- Create: `talon_rl/tasks/locomotion/__init__.py`
- Create: `talon_rl/tasks/locomotion/a1_env/__init__.py`
- Create: `talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py`
- Create: `talon_rl/tasks/locomotion/a1_env/a1_env.py`
- Create: `talon_rl/tasks/locomotion/a1_env/mdp/__init__.py`
- Create: `talon_rl/tasks/locomotion/a1_env/mdp/observations.py`
- Create: `talon_rl/tasks/locomotion/a1_env/mdp/terminations.py`
- Test: `tests/test_a1_env.py`

**Interfaces:**
- Consumes: `talon_rl.config.{ObservationSpaceCfg, ActionSpaceCfg}`,
  `talon_rl.reward.compute_reward_vector`, `talon_rl.envs.base_env.BaseTalonEnv`,
  `talon_rl.config.RewardVectorCfg` (for `progress_std`). Actuator group
  `"base_legs"`, `Kp=55.0`/`Kd=0.8`, `num_envs` from Task 1 — all Global
  Constraints values.
- Produces: `talon_rl.assets.a1.TALON_A1_CFG` (an `ArticulationCfg`),
  `Isaac-Talon-A1-v0` gym task. `gym.make("Isaac-Talon-A1-v0")`
  returns an `IsaacLabTalonEnv` satisfying `BaseTalonEnv` directly (`reset() -> dict`,
  `step(action: (N, 12)) -> tuple[dict, (N,) bool]`) — consumed by Task 8
  (`train_prelim.py`).

**No `mdp/rewards.py` or `mdp/events.py` files** (a deliberate deviation from
a literal jaykorea 1:1 mirror, decided while writing this task — see
`a1_env_cfg.py`'s module docstring below): `RewardsCfg` stays empty and `IsaacLabTalonEnv.step()` calls
`talon_rl.reward.compute_reward_vector()` directly on the transition dict it
already has to build anyway, rather than a separate layer of
manager-term-shaped functions that would just re-extract the same fields a
second time for no functional benefit (nothing registers them with
`RewardManager`, since that manager can only sum to a scalar — see Global
Constraints). Similarly, the reset event needs no custom function —
`isaaclab.envs.mdp.reset_scene_to_default` (re-exported through
`mdp/__init__.py`) is referenced directly from `a1_env_cfg.py`.

- [ ] **Step 1: Remove the 2026-09-13 single-env implementation**

```bash
git rm talon_rl/envs/isaac_lab_env.py tests/test_isaac_lab_env.py
```

- [ ] **Step 2: Scaffold the package**

```bash
mkdir -p talon_rl/tasks/locomotion/a1_env/mdp talon_rl/assets
touch talon_rl/tasks/__init__.py
touch talon_rl/tasks/locomotion/__init__.py
touch talon_rl/assets/__init__.py
```

- [ ] **Step 3: Write `talon_rl/assets/a1.py`**

Mirrors `jaykorea/Isaac-RL-Two-wheel-Legged-Bot`'s `assets/<robot>/*.py`
convention (one named, reusable `ArticulationCfg` constant per file) —
pulls the Kp/Kd override out of `a1_env_cfg.py`'s `__post_init__` so it has
a name and can be imported/reused on its own, instead of being buried
inline in the env cfg.

```python
# talon_rl/assets/a1.py
"""Talon's Unitree A1 config — UNITREE_A1_CFG with the actuator gains
overridden to RMA's (Kumar et al. 2021) Kp=55/Kd=0.8, confirmed against the
real installed UNITREE_A1_CFG's actuator group key ("base_legs", not the
first guess of "legs") on 2026-09-13/14. See
docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md's Risks
section for why RMA's value was chosen over legged_gym's.
"""

from __future__ import annotations

from isaaclab_assets import UNITREE_A1_CFG

_A1_ACTUATOR_GROUP = "base_legs"
_A1_KP = 55.0
_A1_KD = 0.8

TALON_A1_CFG = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
TALON_A1_CFG.actuators[_A1_ACTUATOR_GROUP].stiffness = _A1_KP
TALON_A1_CFG.actuators[_A1_ACTUATOR_GROUP].damping = _A1_KD
```

- [ ] **Step 4: Write `mdp/observations.py`**

```python
# talon_rl/tasks/locomotion/a1_env/mdp/observations.py
"""Observation terms for IsaacLabTalonEnv. Reuses Isaac Lab's own
isaaclab.envs.mdp.joint_pos, .joint_vel, .last_action for those fields (see
mdp/__init__.py's re-export) — roll_pitch, foot_contact_binary, and
v_command have no Isaac Lab builtin matching ObservationSpaceCfg's exact
shape/meaning, so they're written here.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def roll_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(roll, pitch) in radians from the root quaternion (w, x, y, z),
    standard aerospace convention — same formula as the single-env
    IsaacLabTalonEnv's _quat_to_roll_pitch helper (2026-09-13), ported to
    batched torch. (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return torch.stack([roll, pitch], dim=-1)


def foot_contact_binary(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"), threshold: float = 1.0
) -> torch.Tensor:
    """(N, 4) binary contact per foot, matching ObservationSpaceCfg.foot_contact_dim."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = torch.norm(sensor.data.net_forces_w, dim=-1)
    return (force_mag > threshold).float()


def v_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Scripted constant forward-velocity command — no real CommandManager
    in this prelim (matches the single-env IsaacLabTalonEnv's
    self.v_command, 2026-09-13). (N, 3)."""
    return env.v_command_buf
```

- [ ] **Step 5: Write `mdp/terminations.py`**

```python
# talon_rl/tasks/locomotion/a1_env/mdp/terminations.py
"""Termination terms. Horizon timeout uses Isaac Lab's own mdp.time_out
builtin (referenced directly in a1_env_cfg.py's TerminationsCfg) — this file
only adds the scripted obstacle check, ported from the single-env
IsaacLabTalonEnv's `pos_x >= obstacle_ahead` condition (2026-09-13), still a
placeholder pending the Exteroception Module.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_reached(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0] >= env.obstacle_ahead_buf
```

- [ ] **Step 6: Write `mdp/__init__.py`**

```python
# talon_rl/tasks/locomotion/a1_env/mdp/__init__.py
"""Term functions for the A1 task — re-exports Isaac Lab's own builtins
(isaaclab.envs.mdp: joint_pos, joint_vel, last_action, time_out,
reset_scene_to_default, JointPositionActionCfg, ...) plus this task's own
observations.py and terminations.py, matching the reference repo's mdp
package convention (jaykorea/Isaac-RL-Two-wheel-Legged-Bot,
isaaclab_tasks' own cartpole/mdp/__init__.py)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import foot_contact_binary, roll_pitch, v_command  # noqa: F401
from .terminations import obstacle_reached  # noqa: F401
```

- [ ] **Step 7: Write `a1_env_cfg.py`**

```python
# talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py
"""IsaacLabTalonEnvCfg(ManagerBasedRLEnvCfg) — scene + observations +
actions + terminations + events, composed from mdp/*.py term functions per
Isaac Lab's manager-based convention (mirrors isaaclab_tasks' own
cartpole_env_cfg.py and jaykorea's velocity_env_cfg.py, both verified
2026-09-14 against the real installed isaaclab 0.48.0).

RewardsCfg is deliberately empty — see a1_env.py's step() override and the
design doc's Decision 2: this repo needs an unsummed 5-term reward vector,
which RewardManager's scalar-sum contract can't produce, so reward
computation happens directly in step() via
talon_rl.reward.compute_reward_vector() instead of through this manager.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from talon_rl.assets.a1 import TALON_A1_CFG
from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg

from . import mdp

# Set by Task 1's empirical VRAM sizing (2026-09-14) — replace this literal
# if Task 1 found a different value fits the RTX 3070 Ti's 8GB better.
_DEFAULT_NUM_ENVS = 2048


@configclass
class A1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = MISSING  # set in IsaacLabTalonEnvCfg.__post_init__

    contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=1)

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        roll_pitch = ObsTerm(func=mdp.roll_pitch)
        foot_contact = ObsTerm(func=mdp.foot_contact_binary)
        last_action = ObsTerm(func=mdp.last_action)
        v_command = ObsTerm(func=mdp.v_command)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    obstacle_reached = DoneTerm(func=mdp.obstacle_reached)


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Deliberately empty — see module docstring."""


@configclass
class IsaacLabTalonEnvCfg(ManagerBasedRLEnvCfg):
    scene: A1SceneCfg = A1SceneCfg(num_envs=_DEFAULT_NUM_ENVS, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # .replace() with no actual changes, not a direct assignment: TALON_A1_CFG
        # is a shared module-level object, and every IsaacLabTalonEnvCfg()
        # instance (e.g. Task 6's own structural test constructs more than
        # one) needs its own copy, or mutating one instance's scene.robot
        # would leak into every other instance sharing the same object.
        self.scene.robot = TALON_A1_CFG.replace()

        self.decimation = 1
        self.episode_length_s = 200 * 0.02  # matches the 2026-09-13 single-env horizon=200, dt=0.02
        self.sim.dt = 0.02
        self.sim.render_interval = self.decimation

        _obs_cfg = ObservationSpaceCfg()
        _action_cfg = ActionSpaceCfg()
        self.obs_dim = _obs_cfg.total_dim - _obs_cfg.preference_dim
        self.action_dim = _action_cfg.dim
```

- [ ] **Step 8: Write `a1_env.py`**

```python
# talon_rl/tasks/locomotion/a1_env/a1_env.py
"""IsaacLabTalonEnv(ManagerBasedRLEnv) — the class registered as
Isaac-Talon-A1-v0. Scene/observations/actions/terminations/events all come
from IsaacLabTalonEnvCfg's manager configs (a1_env_cfg.py) — this class
only adds: (1) the v_command/obstacle_ahead scripted buffers
mdp/observations.py and mdp/terminations.py read, and (2) step()/reset()
overrides that return this repo's BaseTalonEnv-shaped
(transition_dict, done_array) instead of gym.Env's raw tuple, since nothing
outside this repo needs generic gym.Env compliance from this class
(train_prelim.py drives it directly via BaseTalonEnv) — see
docs/superpowers/specs/2026-09-14-vectorized-isaac-lab-env-design.md.
"""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.reward import compute_reward_vector

from .a1_env_cfg import IsaacLabTalonEnvCfg


class IsaacLabTalonEnv(ManagerBasedRLEnv, BaseTalonEnv):
    cfg: IsaacLabTalonEnvCfg

    def __init__(self, cfg: IsaacLabTalonEnvCfg, reward_cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.num_envs = self.cfg.scene.num_envs
        self.obs_dim = self.cfg.obs_dim
        self.action_dim = self.cfg.action_dim
        self.v_command_buf = torch.tensor([0.5, 0.0, 0.0], device=self.device).expand(self.num_envs, 3).contiguous()
        self.obstacle_ahead_buf = torch.full((self.num_envs,), 5.0, device=self.device)
        from talon_rl.config import RewardVectorCfg
        self._reward_cfg = reward_cfg or RewardVectorCfg()

    def reset(self, **kwargs) -> dict:
        obs_dict, _extras = super().reset(**kwargs)
        return self._transition(obs_dict)

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        action_t = torch.from_numpy(action).to(self.device)
        obs_dict, _reward_buf, terminated, truncated, _extras = super().step(action_t)
        transition = self._transition(obs_dict)
        done = (terminated | truncated).cpu().numpy()
        return transition, done

    def _transition(self, obs_dict: dict) -> dict:
        robot = self.scene["robot"]
        prev_action = self.action_manager.prev_action.cpu().numpy()
        action = self.action_manager.action.cpu().numpy()
        transition = {
            "obs": obs_dict["policy"].cpu().numpy().astype(np.float32),
            "v_actual": robot.data.root_lin_vel_b.cpu().numpy(),
            "v_command": self.v_command_buf.cpu().numpy(),
            "obstacle_dist": np.maximum(
                0.0, (self.obstacle_ahead_buf - robot.data.root_pos_w[:, 0]).cpu().numpy()
            ),
            "joint_torque": robot.data.applied_torque.cpu().numpy(),
            "joint_vel": robot.data.joint_vel.cpu().numpy(),
            "joint_acc": (action - prev_action) / self.step_dt,
            "foot_contact_force": torch.norm(
                self.scene.sensors["contact_sensor"].data.net_forces_w, dim=-1
            ).cpu().numpy(),
            "action": action,
            "prev_action": prev_action,
        }
        transition["reward_vec"] = compute_reward_vector(transition, self._reward_cfg)
        return transition
```

`reset()` needs no special-casing: `ActionManager` starts every lane's
`prev_action`/`action` at zero on construction and on every reset, so
`_transition()`'s `obs_dict["policy"]` (which includes `last_action` via
the observation manager) is already correctly zeroed on the first frame
without any extra handling here.

- [ ] **Step 9: Write `__init__.py` (gym registration)**

```python
# talon_rl/tasks/locomotion/a1_env/__init__.py
import gymnasium as gym

from .a1_env import IsaacLabTalonEnv
from .a1_env_cfg import IsaacLabTalonEnvCfg

gym.register(
    id="Isaac-Talon-A1-v0",
    entry_point=IsaacLabTalonEnv,
    disable_env_checker=True,  # our step()/reset() return this repo's own
                                # (transition_dict, done_array) shape, not
                                # gym's standard tuple — the checker would
                                # reject that as non-conformant.
    kwargs={"cfg": IsaacLabTalonEnvCfg()},
)
```

- [ ] **Step 10: Write the structural test**

```python
# tests/test_a1_env.py
"""Structural contract test for IsaacLabTalonEnv — skips entirely on a venv
without Isaac Sim installed (e.g. the repo's default 3.12 .venv). The real
proof this env works at scale is the manual GPU smoke run documented in
README, not this test.
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import pytest

pytest.importorskip("isaacsim")

import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.envs.base_env import BaseTalonEnv


def test_isaac_lab_env_implements_base_contract():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0

        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = 4  # small N for a fast structural check
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

        obs_cfg = ObservationSpaceCfg()
        action_cfg = ActionSpaceCfg()
        assert isinstance(env, BaseTalonEnv)
        assert env.num_envs == 4
        assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
        assert env.action_dim == action_cfg.dim

        transition = env.reset()
        assert transition["obs"].shape == (4, env.obs_dim)
        for key in (
            "v_actual", "v_command", "obstacle_dist", "joint_torque", "joint_vel",
            "foot_contact_force", "action", "prev_action", "joint_acc",
        ):
            assert key in transition
            assert transition[key].shape[0] == 4

        action = np.zeros((4, env.action_dim), dtype=np.float32)
        transition, done = env.step(action)
        assert transition["obs"].shape == (4, env.obs_dim)
        assert done.shape == (4,)
        assert np.all(np.isfinite(transition["obs"]))
        assert np.all(np.isfinite(transition["reward_vec"]))
    finally:
        import threading
        watchdog = threading.Timer(15.0, lambda: os._exit(0))
        watchdog.daemon = True
        watchdog.start()
        app.close()
        watchdog.cancel()
```

- [ ] **Step 11: Run the structural test**

Run:
```bash
source ~/isaac-lab-env/bin/activate
cd "/home/xero/Master's Degree/Thesis/talon-rl"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 280 pytest tests/test_a1_env.py -v
```
Expected: exit code 0. If it fails with an `AttributeError`/`KeyError`
naming a manager attribute (`action_manager`, `scene.sensors`, etc.), it
means the real installed isaaclab 0.48.0 API differs from what
`~/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py` and
`.../envs/manager_based_env.py` showed during this plan's writing
(2026-09-14) — re-grep those files for the current attribute name and fix
`a1_env.py`'s `_transition()` accordingly; do not guess a second time.

- [ ] **Step 12: Run the 3.12 `.venv` to confirm it skips cleanly**

Run:
```bash
cd "/home/xero/Master's Degree/Thesis/talon-rl"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_a1_env.py -v
```
Expected: `SKIPPED (could not import 'isaacsim')`.

- [ ] **Step 13: Commit**

```bash
git add talon_rl/assets talon_rl/tasks tests/test_a1_env.py
git add -u talon_rl/envs tests/test_isaac_lab_env.py  # stages the deletions from Step 1
git commit -m "feat: rewrite IsaacLabTalonEnv onto ManagerBasedRLEnv + gym.register

Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's tasks/manager_based/
locomotion/velocity/<robot>_env/ + mdp/ + assets/<robot>.py layout.
RewardsCfg stays empty — step() calls
talon_rl.reward.compute_reward_vector() directly for the unsummed 5-term
vector MOPPO needs, which RewardManager's scalar-sum contract can't
produce."
```

---

### Task 7: `training/moppo.py` — persistent rollout, done-masked GAE

**Files:**
- Modify: `talon_rl/training/moppo.py` (full rewrite)
- Modify: `tests/test_moppo_smoke.py` (full rewrite)

**Interfaces:**
- Consumes: `BaseTalonEnv` (Task 2), `compute_reward_vector` (Task 3),
  `sample_preference_vector`/`rate_limit`/`floor_clip` (Task 4),
  `ObservationStack` (Task 5).
- Produces: `MOPPOConfig(num_steps: int = 24, ...)` (replaces
  `episodes_per_update`), `MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg, stack_cfg, seed).update() -> dict`
  (same stats-dict keys as before: `policy_loss`, `value_loss`,
  `mean_reward_vec`, `mean_episode_len`). Consumed by Task 8
  (`train_prelim.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_moppo_smoke.py
"""End-to-end smoke test: does the whole MOPPO loop run on a vectorized
DummyTalonEnv without error, and do the reported numbers stay finite? This
is NOT a convergence test — the dummy env has no locomotion physics, so
"the policy got better" isn't a meaningful claim here. It only proves
obs -> policy(w) -> action -> reward vector -> vector critic -> PPO update
is wired correctly end to end, now with N parallel lanes and persistent
rollout collection across update() calls.
"""

import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
from talon_rl.envs.dummy_env import DummyTalonEnv
from talon_rl.training.moppo import MOPPOConfig, MOPPOTrainer


def test_moppo_runs_a_few_updates_without_nans():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=8, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=10, epochs_per_update=2),
        seed=0,
    )

    for _ in range(3):
        stats = trainer.update()
        assert np.isfinite(stats["policy_loss"])
        assert np.isfinite(stats["value_loss"])
        assert np.all(np.isfinite(stats["mean_reward_vec"]))
        assert stats["mean_reward_vec"].shape == (reward_cfg.dim,)


def test_moppo_runs_with_mismatched_actor_critic_stacks():
    """Exercises the Flamingo-style num_policy_stacks != num_critic_stacks path —
    actor sees 4 frames of history, critic sees only the current frame."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg(num_policy_stacks=4, num_critic_stacks=1)

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=10, epochs_per_update=2),
        stack_cfg=stack_cfg,
        seed=0,
    )

    stats = trainer.update()
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


def test_rollout_is_persistent_across_update_calls():
    """The rollout must continue where the previous update() left off, not
    reset every call — regression test for the episode-based -> persistent
    rollout behavioral change (see design doc's moppo.py section)."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=2, horizon=1000, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        seed=0,
    )
    t_before = trainer._t  # internal step counter this task's implementation must expose
    trainer.update()
    t_after = trainer._t
    assert t_after == t_before + 5  # advanced by exactly num_steps, not reset to 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_moppo_smoke.py -v`
Expected: FAIL (`TypeError` — `DummyTalonEnv` positional args changed,
`MOPPOConfig` has no `num_steps`).

- [ ] **Step 3: Rewrite `moppo.py`**

```python
# talon_rl/training/moppo.py
"""Minimal preference-conditioned PPO (MOPPO), chapter3.tex §3.2.3 / fig 3.3.

Policy: pi(a | s, w) — observation is [s, w] concatenated (w appended by this
module, not by the env — see envs/base_env.py docstring). The actor and
critic can see *different* amounts of temporal history via
`ObservationStackCfg` (num_policy_stacks / num_critic_stacks, after Flamingo
— jaykorea/Isaac-RL-Two-wheel-Legged-Bot — see obs_stack.py).

Critic: vector critic V(s, w) -> R^5, one head per reward-vector term
(asymmetric actor-critic per AMOR \\cite{alegre2025}).
Policy-gradient advantage: scalarized as w . advantage_vector, i.e. the
preference vector arbitrates between objectives at the advantage level, not
by pre-summing the reward into a scalar before GAE.

Rollout collection is fixed-horizon + auto-reset (Isaac Lab/rsl_rl/sb3-
standard, 2026-09-14) — N env lanes step in lockstep for `num_steps` per
update() call, any lane that terminates auto-resets internally and keeps
contributing to the same buffer, and the rollout is PERSISTENT: env.reset()
happens once (in __init__), and each update() call collects the next
num_steps timesteps continuing wherever the previous call left off (not a
fresh episode-based rollout every call, unlike the pre-2026-09-14 version).
GAE uses the standard done-masked recursion so value bootstrapping never
crosses an episode boundary within a lane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..config import ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
from ..envs.base_env import BaseTalonEnv
from ..obs_stack import ObservationStack
from ..preference import floor_clip, rate_limit, sample_preference_vector
from ..reward import compute_reward_vector


@dataclass
class MOPPOConfig:
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    num_steps: int = 24  # rollout length per update() call, across all N lanes
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, reward_dim: int, hidden_dim: int):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(actor_obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic_body = nn.Sequential(
            nn.Linear(critic_obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden_dim, reward_dim)

    def act(self, actor_obs_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        dist = Normal(mean, self.log_std.exp())
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp

    def logp(self, actor_obs_w: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        dist = Normal(mean, self.log_std.exp())
        return dist.log_prob(action).sum(-1)

    def value(self, critic_obs_w: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(critic_obs_w))


def _gae_per_objective(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float
) -> np.ndarray:
    """rewards: (T, N, K), values: (T+1, N, K), dones: (T, N) -> advantages (T, N, K).
    dones[t] masks out value bootstrapping across an episode boundary at step t."""
    T, N, K = rewards.shape
    adv = np.zeros((T, N, K), dtype=np.float32)
    gae = np.zeros((N, K), dtype=np.float32)
    for t in reversed(range(T)):
        mask = (1.0 - dones[t])[:, None]  # (N, 1), broadcasts over K
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    return adv


class MOPPOTrainer:
    def __init__(
        self,
        env: BaseTalonEnv,
        obs_cfg: ObservationSpaceCfg,
        reward_cfg: RewardVectorCfg,
        pref_cfg: PreferenceCfg,
        moppo_cfg: MOPPOConfig | None = None,
        stack_cfg: ObservationStackCfg | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.pref_cfg = pref_cfg
        self.cfg = moppo_cfg or MOPPOConfig()
        self.stack_cfg = stack_cfg or ObservationStackCfg()
        self.rng = np.random.default_rng(seed)
        self.n = env.num_envs

        self.stack = ObservationStack(
            self.n, env.obs_dim, self.stack_cfg.num_policy_stacks, self.stack_cfg.num_critic_stacks
        )
        actor_obs_w_dim = self.stack.policy_obs_dim + reward_cfg.dim
        critic_obs_w_dim = self.stack.critic_obs_dim + reward_cfg.dim
        self.model = ActorCritic(actor_obs_w_dim, critic_obs_w_dim, env.action_dim, reward_cfg.dim, self.cfg.hidden_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        # Persistent rollout state — set up once here, advanced by update(),
        # never reset mid-training (auto-reset happens per-lane inside
        # env.step() itself).
        transition = self.env.reset()
        self.stack.reset(transition["obs"])
        self.w = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, self.n)
        self.w = floor_clip(self.w, self.reward_cfg.term_names, self.reward_cfg.impact_floor_eps)
        self._prev_done = np.zeros(self.n, dtype=bool)
        self._t = 0

    def _collect_rollout(self) -> dict:
        actor_obs_list, critic_obs_list, act_list, logp_list, rew_list, val_list, done_list = (
            [], [], [], [], [], [], []
        )

        for _ in range(self.cfg.num_steps):
            # Lanes that just auto-reset (done from the PREVIOUS step) get a
            # fresh w sample (new episode -> new preference sample, fig 3.3);
            # still-running lanes rate-limit toward a freshly resampled
            # target, exactly as the pre-2026-09-14 per-episode version did.
            w_target = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, self.n)
            w_rate_limited = rate_limit(self.w, w_target, self.pref_cfg.max_delta_per_step)
            self.w = np.where(self._prev_done[:, None], w_target, w_rate_limited)
            self.w = floor_clip(self.w, self.reward_cfg.term_names, self.reward_cfg.impact_floor_eps)

            actor_obs_w = np.concatenate([self.stack.policy_obs, self.w], axis=-1).astype(np.float32)
            critic_obs_w = np.concatenate([self.stack.critic_obs, self.w], axis=-1).astype(np.float32)
            with torch.no_grad():
                action_t, logp_t = self.model.act(torch.from_numpy(actor_obs_w))
                value_t = self.model.value(torch.from_numpy(critic_obs_w))
            action = action_t.numpy()

            transition, done = self.env.step(action)
            self.stack.push(transition["obs"], done_mask=done)
            reward_vec = compute_reward_vector(transition, self.reward_cfg)

            actor_obs_list.append(actor_obs_w)
            critic_obs_list.append(critic_obs_w)
            act_list.append(action)
            logp_list.append(logp_t.numpy())
            rew_list.append(reward_vec)
            val_list.append(value_t.numpy())
            done_list.append(done)

            self._prev_done = done
            self._t += 1

        with torch.no_grad():
            final_critic_obs_w = np.concatenate([self.stack.critic_obs, self.w], axis=-1).astype(np.float32)
            final_value = self.model.value(torch.from_numpy(final_critic_obs_w)).numpy()

        return {
            "actor_obs": np.stack(actor_obs_list),      # (T, N, actor_obs_w_dim)
            "critic_obs": np.stack(critic_obs_list),    # (T, N, critic_obs_w_dim)
            "actions": np.stack(act_list),               # (T, N, action_dim)
            "logp": np.stack(logp_list),                 # (T, N)
            "rewards": np.stack(rew_list),                # (T, N, K)
            "values": np.stack(val_list),                 # (T, N, K)
            "dones": np.stack(done_list),                  # (T, N)
            "final_value": final_value,                     # (N, K)
        }

    def update(self) -> dict:
        """Collects the next `num_steps` timesteps (continuing the persistent
        rollout), then runs PPO for `epochs_per_update` epochs."""
        r = self._collect_rollout()

        values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)  # (T+1, N, K)
        adv = _gae_per_objective(r["rewards"], values_with_final, r["dones"], self.cfg.gamma, self.cfg.gae_lambda)
        returns = adv + r["values"]

        T, N = r["dones"].shape
        # w . advantage_vector per (t, n) — the preference vector arbitrates
        # between objectives at the advantage level (not a pre-summed scalar
        # reward before GAE). w is the last reward_cfg.dim columns of the
        # stored actor_obs (see the w-concatenation in _collect_rollout).
        w_used = r["actor_obs"][:, :, -self.reward_cfg.dim :]
        scalar_adv = np.einsum("tnk,tnk->tn", adv, w_used).reshape(T * N)

        actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1))
        critic_obs_t = torch.from_numpy(r["critic_obs"].reshape(T * N, -1))
        actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1))
        logp_old_t = torch.from_numpy(r["logp"].reshape(T * N))
        adv_t = torch.from_numpy(scalar_adv.astype(np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = torch.from_numpy(returns.reshape(T * N, -1).astype(np.float32))

        last_policy_loss = last_value_loss = 0.0
        for _ in range(self.cfg.epochs_per_update):
            logp_new = self.model.logp(actor_obs_t, actions_t)
            ratio = torch.exp(logp_new - logp_old_t)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            values_pred = self.model.value(critic_obs_t)
            value_loss = nn.functional.mse_loss(values_pred, returns_t)

            loss = policy_loss + 0.5 * value_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            last_policy_loss, last_value_loss = float(policy_loss.item()), float(value_loss.item())

        mean_reward_vec = r["rewards"].reshape(T * N, -1).mean(axis=0)
        # mean episode length across lanes that actually finished an episode
        # during this rollout window; falls back to num_steps if none did
        # (a short num_steps relative to horizon).
        finished_lengths = []
        for lane in range(N):
            lane_dones = np.where(r["dones"][:, lane])[0]
            if lane_dones.size:
                finished_lengths.extend(np.diff(np.concatenate([[-1], lane_dones])))
        mean_episode_len = float(np.mean(finished_lengths)) if finished_lengths else float(T)

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "mean_reward_vec": mean_reward_vec,
            "mean_episode_len": mean_episode_len,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_moppo_smoke.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full local suite to confirm nothing else broke**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/ -v --ignore=tests/test_a1_env.py`
Expected: all pass (Tasks 2-5's tests plus this task's).

- [ ] **Step 6: Commit**

```bash
git add talon_rl/training/moppo.py tests/test_moppo_smoke.py
git commit -m "feat: persistent fixed-horizon rollout + done-masked GAE in moppo.py

Replaces per-update() episode collection with a continuous rollout (env
resets once, each update() collects the next num_steps across all N
lanes) — the rsl_rl/sb3-standard vectorized-PPO pattern."
```

---

### Task 8: `scripts/train_prelim.py` — `--num_envs` wiring

**Files:**
- Modify: `scripts/train_prelim.py`

**Interfaces:**
- Consumes: `DummyTalonEnv(obs_cfg, action_cfg, num_envs, horizon, seed)`
  (Task 2); `gym.make("Isaac-Talon-A1-v0", cfg=...)` (Task 6).

- [ ] **Step 1: Add the flag**

```python
# scripts/train_prelim.py — replace the body of main()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=2048)  # Task 1's empirically-sized default
    args = parser.parse_args()

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    if args.env == "dummy":
        env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=args.num_envs, horizon=200, seed=args.seed)
    else:
        # Imported lazily so --env dummy keeps working on machines without
        # Isaac Sim installed (this repo's default 3.12 .venv included).
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), seed=args.seed)

    print(f"reward terms: {reward_cfg.term_names}")
    for i in range(1, args.updates + 1):
        stats = trainer.update()
        r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, stats["mean_reward_vec"]))
        print(
            f"update {i:3d} | policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} ep_len={stats['mean_episode_len']:.1f} | {r}"
        )

    if args.env == "isaac_lab":
        import threading
        watchdog = threading.Timer(15.0, lambda: __import__("os")._exit(0))
        watchdog.daemon = True
        watchdog.start()
        env.close()
        watchdog.cancel()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the dummy path still works (now with real parallelism)**

Run:
```bash
cd "/home/xero/Master's Degree/Thesis/talon-rl"
.venv/bin/python scripts/train_prelim.py --updates 3 --num_envs 8
```
Expected: 3 update lines print, `ep_len` no longer a fixed round number
every time (varies with which lanes happened to finish during each
`num_steps`-long window) — this is expected given the persistent-rollout
behavioral change (Task 7), not a bug.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_prelim.py
git commit -m "feat: add --num_envs flag, wire IsaacLabTalonEnv via gym.make"
```

---

### Task 9: Manual GPU smoke test and README update

**Files:**
- Modify: `README.md`

**Interfaces:** none (this task's deliverable is a verified manual run plus
a documented command, not new code).

- [ ] **Step 1: Run the real smoke test**

```bash
source ~/isaac-lab-env/bin/activate
cd "/home/xero/Master's Degree/Thesis/talon-rl"
timeout 280 python scripts/train_prelim.py --env isaac_lab --updates 5 --num_envs <Task 1's chosen value>
```
Isaac Sim's stdout gets swallowed by Kit's own logging takeover once fully
booted (confirmed 2026-09-14) — printed update lines will not be visible in
the terminal. To get visible proof, write a one-off variant to a file
instead (same pattern used to verify the single-env version on
2026-09-14): copy `main()`'s loop into a `/tmp/smoke_proof.py` that writes
each `update N | ...` line to `/tmp/smoke_proof_output.txt` with `f.flush()`
after each line, run it the same way, then `cat /tmp/smoke_proof_output.txt`.
Expected: 5 update lines with finite numbers, same shape as the dummy env's
output. This is the actual proof this milestone's goal is met — per the
design doc's non-goals, do **not** interpret the printed reward values as a
locomotion result of any kind, only as "the loop completed at num_envs
scale."

- [ ] **Step 2: Document the manual step in README**

Find the existing `### Isaac Lab smoke test` section (added 2026-09-13) and
replace its command block:

```markdown
### Isaac Lab smoke test (requires the separate `~/isaac-lab-env` venv, GPU machine only)

```bash
source ~/isaac-lab-env/bin/activate
python scripts/train_prelim.py --env isaac_lab --updates 5 --num_envs <chosen value>
```

Proves the pipeline runs against a real, vectorized Isaac Lab environment
(`ManagerBasedRLEnv`, `<chosen value>` parallel A1 clones) — same "doesn't
crash" bar as the single-env 2026-09-13 version, not a locomotion result.
See `docs/superpowers/specs/2026-09-14-vectorized-isaac-lab-env-design.md`
for the vectorization design and
`docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md` for the
original single-env setup and known limitations.
```

- [ ] **Step 3: Update "Next milestones"**

Mark the vectorization work done, matching the existing strikethrough style
from 2026-09-13:

```markdown
1. ~~Write `IsaacLabTalonEnv(BaseTalonEnv)` against the real Unitree A1 asset.~~ Done — see `talon_rl/tasks/locomotion/a1_env/`.
```

(No new milestone line needed — vectorization was this repo's own
follow-up to milestone 1, not a separately tracked milestone.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document the vectorized Isaac Lab smoke-test manual step"
```

---

## Plan self-review notes

- **Spec coverage:** Design doc's Decision 1 (fixed-horizon + auto-reset) ->
  Task 7. Decision 2 (`ManagerBasedRLEnv` + `gym.register`, vector-reward
  bypass) -> Task 6. Decision 3 (`BaseTalonEnv` batch-native,
  `gym.vector.SyncVectorEnv` for Dummy) -> Task 2. Decision 4 (`num_envs`
  empirical sizing from 2048) -> Task 1, consumed by Tasks 6 and 8.
  `reward.py`/`preference.py`/`obs_stack.py` batching -> Tasks 3-5.
  `scripts/train_prelim.py` `--num_envs` -> Task 8. Manual GPU proof + README
  -> Task 9.
- **Design doc deviation, recorded here since it happened while writing this
  plan, not during brainstorming:** the design doc described `mdp/rewards.py`
  and `mdp/events.py` as files mirroring jaykorea's folder shape. Verifying
  against the real installed isaaclab 0.48.0 (2026-09-14, reading
  `ManagerBasedRLEnv.step()`'s actual reward-computation call site and
  `isaaclab.envs.mdp.events.reset_scene_to_default`'s builtin) showed both
  would either duplicate `talon_rl/reward.py`'s math for no functional
  reason or contain no real code at all. Task 6 omits both files and
  documents why in its own header instead of manufacturing empty/duplicate
  files just for folder-shape parity — the substantive part of the parity
  goal (the `tasks/locomotion/a1_env/` + `mdp/` layout, `ManagerBasedRLEnv`
  itself, reusing Isaac Lab's builtin `joint_pos`/`joint_vel`/`last_action`/
  `time_out`/`reset_scene_to_default` terms) is intact.
- **Known deliberate gap carried over from 2026-09-13:** `obstacle_dist`/
  `clearance_reward` is still a scripted placeholder (no Exteroception
  Module) — Task 6's `terminations.py`/`a1_env.py` keep this explicit, not
  silently dropped.
- **Type consistency check:** `BaseTalonEnv.step()` return type
  (`tuple[dict, np.ndarray]`, `done` boolean) is identical across Task 2's
  contract, Task 2's `DummyTalonEnv`, Task 6's `IsaacLabTalonEnv`, and Task
  7's `MOPPOTrainer._collect_rollout()` usage. `ObservationStack`'s
  constructor argument order (`num_envs, obs_dim, ...`) is consistent
  between Task 5's definition and Task 7's `MOPPOTrainer.__init__` call.
  `MOPPOConfig.num_steps` (not `episodes_per_update`) is used consistently
  in Task 7's implementation, tests, and Task 8's `train_prelim.py` (which
  doesn't reference either field directly, only constructs `MOPPOConfig()`
  with defaults).
