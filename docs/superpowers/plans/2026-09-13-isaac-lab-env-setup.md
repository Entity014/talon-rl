# Isaac Lab Environment Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `IsaacLabTalonEnv(BaseTalonEnv)`, a real (non-dummy) Isaac
Lab environment for the Unitree A1, and prove the existing MOPPO pipeline
runs against it headless without crashing — the same "pipeline runs" bar
`DummyTalonEnv` already clears, not a locomotion claim.

**Architecture:** `IsaacLabTalonEnv` wraps a single-articulation Isaac Lab
scene directly (`SimulationContext` + `Articulation` + `ContactSensor`, the
lower-level pattern Isaac Lab's own "Interacting with an articulation"
tutorial uses) rather than Isaac Lab's full gym-style `DirectRLEnv` manager
machinery — `BaseTalonEnv`'s `reset()`/`step()` contract is a single
non-vectorized transition dict, simpler than what `DirectRLEnv` returns, and
`training/moppo.py` is already the training loop, so we don't need Isaac
Lab's own MDP managers. `num_envs=1` for this milestone (matches the RTX
3070 Ti VRAM ceiling already flagged as a known risk).

**Tech Stack:** Isaac Sim 5.X (pip), Isaac Lab `release/2.3.0`, Python 3.11,
`uv`-managed venv at `~/isaac-lab-env`, PyTorch (ships with Isaac Sim).

**Spec:** `docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md`

## Global Constraints

- Isaac Sim version: **5.X** (pip package `isaacsim[all]`)
- Isaac Lab version: tag **`release/2.3.0`** (not `main`/beta)
- Venv: `~/isaac-lab-env`, Python **3.11 exactly** (Isaac Sim 5.X hard
  requirement) — separate from `talon-rl/.venv` (3.12, dummy-env pytest)
- `training/moppo.py` must not change at all — `IsaacLabTalonEnv` satisfies
  the exact same `BaseTalonEnv` contract `DummyTalonEnv` does
- `obs_dim` **excludes** the preference vector $w$ — never append it inside
  the env (moppo.py does this)
- PD gains: use RMA's $K_p=55$, $K_d=0.8$ (this repo's own Adaptation Module
  reference paper), overriding Isaac Lab's shipped `UNITREE_A1_CFG` default
  ($K_p=25.0$, $K_d=0.5$) explicitly — don't silently keep Isaac Lab's default
- No locomotion-result claims anywhere (README's existing rule) — "doesn't
  crash" is the only bar this milestone targets

---

### Task 1: Environment setup and installation verification

**Files:** none in this repo — all steps are shell commands against the host
machine and `~/isaac-lab-env`.

**Interfaces:**
- Produces: a working `~/isaac-lab-env` venv with `isaacsim` and `isaaclab`
  importable, and `talon_rl` installed editable into it — everything later
  tasks run inside.

- [ ] **Step 1: Create the venv**

Run:
```bash
uv venv ~/isaac-lab-env --python 3.11
```
Expected: creates `~/isaac-lab-env` with a `bin/python` reporting Python 3.11.x.

- [ ] **Step 2: Install Isaac Sim**

Run:
```bash
source ~/isaac-lab-env/bin/activate
uv pip install "isaacsim[all]==5.*"
```
Expected: completes without error (large download, ~15-25GB). If GLIBC/pip
compatibility errors appear, stop — Ubuntu 24.04 was already confirmed
compatible in the design doc's Context section, so an error here means
something changed and needs investigation, not a silent fallback to the
binary installer.

- [ ] **Step 3: Verify Isaac Sim launches headless**

Run (inside the activated venv):
```bash
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('Isaac Sim launched OK')
app.close()
"
```
Expected: prints `Isaac Sim launched OK` and exits cleanly (may take 1-2
minutes on first launch while it compiles shader caches). If this fails,
stop here — do not proceed to Isaac Lab install until this passes, per the
design doc's explicit isolation of "Isaac Sim installed correctly" from
"our env code is correct."

- [ ] **Step 4: Clone and install Isaac Lab**

Run:
```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout release/2.3.0
./isaaclab.sh --install
```
Run this with `~/isaac-lab-env` activated so `isaaclab.sh` installs into
that venv, not a new one it creates itself — check `isaaclab.sh --help`
first if unsure whether it respects an already-active venv; if it insists on
creating its own, activate `~/isaac-lab-env` again afterward and
`uv pip install -e ~/IsaacLab/source/isaaclab_tasks` (and any other
`isaaclab_*` subpackages `--install` reports) so everything lands in one venv.

Expected: completes without error; `python -c "import isaaclab"` and
`python -c "import isaaclab_assets"` both succeed afterward.

- [ ] **Step 5: Install talon-rl into the same venv**

Run:
```bash
uv pip install -e "/home/xero/Master's Degree/Thesis/talon-rl"
```
Expected: `python -c "import talon_rl; print(talon_rl.__file__)"` prints a
path inside the repo (editable install).

- [ ] **Step 6: Record the exact installed versions**

Run:
```bash
python -c "
import isaacsim, isaaclab
print('isaacsim:', getattr(isaacsim, '__version__', 'unknown'))
print('isaaclab:', getattr(isaaclab, '__version__', 'unknown'))
"
```
Append the output to `docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md`
under a new `## Installed versions (YYYY-MM-DD)` heading — the design doc
named target versions before install; this closes the loop with what
actually landed, since patch versions can shift what's available.

- [ ] **Step 7: Commit the version record**

```bash
cd "/home/xero/Master's Degree/Thesis/talon-rl"
git add docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md
git commit -m "docs: record installed Isaac Sim/Isaac Lab versions"
```

---

### Task 2: Inspect the shipped A1 config and a reference locomotion env

Do this *before* writing `IsaacLabTalonEnv` — it turns two real unknowns
(the A1 actuator group's dict key, and the exact joint-name ordering) into
verified facts instead of guesses baked into Task 3's code.

**Files:**
- Create: `/tmp/inspect_a1.py` (throwaway, not committed)

**Interfaces:**
- Produces: `actuator_group_name: str` (the dict key of `UNITREE_A1_CFG`'s
  single actuator group) and `joint_names: list[str]` (order of the 12
  actuated joints as Isaac Lab enumerates them) — Task 3 uses both.

- [ ] **Step 1: Write and run the inspection script**

```python
# /tmp/inspect_a1.py
from isaaclab_assets import UNITREE_A1_CFG

actuators = UNITREE_A1_CFG.actuators
print("actuator group keys:", list(actuators.keys()))
for name, cfg in actuators.items():
    print(f"  {name}: stiffness={cfg.stiffness} damping={cfg.damping} "
          f"joint_names_expr={cfg.joint_names_expr}")
print("init_state.joint_pos:", UNITREE_A1_CFG.init_state.joint_pos)
```

Run:
```bash
source ~/isaac-lab-env/bin/activate
python /tmp/inspect_a1.py
```

Expected: prints one actuator group (per the design doc's research, a
single `DCMotorCfg` group covering all `.*_hip_joint`, `.*_thigh_joint`,
`.*_calf_joint` — but confirm the exact printed dict key name here, don't
assume it's literally `"legs"` or `"base_legs"`).

- [ ] **Step 2: Write down the actual joint order**

Isaac Lab's `Articulation.data.joint_pos` orders joints by USD-file
discovery order, not necessarily `[FL, FR, RL, RR] x [hip, thigh, calf]`.
Confirm by launching a trivial scene and printing `robot.joint_names`:

```python
# append to /tmp/inspect_a1.py, or run separately after Step 1
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets import UNITREE_A1_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02))
robot_cfg = UNITREE_A1_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)
sim.reset()
print("joint_names:", robot.joint_names)
app.close()
```

Expected: prints a list of 12 joint name strings. Record this list — Task 3
needs it to know which index in `robot.data.joint_pos[0]` maps to which
`ObservationSpaceCfg.joint_pos_dim` slot (order doesn't need to match A1's
physical FL/FR/RL/RR layout exactly, since nothing downstream depends on
*which* index is which leg — `reward.py`'s terms are all order-agnostic
sums/norms over the full 12-vector — but it must be **consistent** between
`joint_pos` and `joint_vel` and `action`, which Step 2 confirms by
construction: all three come from the same `robot.joint_names` ordering).

- [ ] **Step 3: No commit for this task** — the findings feed directly into
  Task 3's code as literal values/comments; the throwaway script itself is
  not part of the repo.

---

### Task 3: Implement `IsaacLabTalonEnv(BaseTalonEnv)`

**Files:**
- Create: `talon_rl/envs/isaac_lab_env.py`
- Test: `tests/test_isaac_lab_env.py`

**Interfaces:**
- Consumes: `BaseTalonEnv` (`talon_rl/envs/base_env.py`) — abstract
  `reset() -> dict`, `step(action: np.ndarray) -> tuple[dict, bool]`;
  `ObservationSpaceCfg`/`ActionSpaceCfg` (`talon_rl/config.py`); the
  actuator group name and joint order recorded in Task 2.
- Produces: `IsaacLabTalonEnv(obs_cfg: ObservationSpaceCfg, action_cfg:
  ActionSpaceCfg, horizon: int = 200, dt: float = 0.02, headless: bool =
  True)` — same constructor shape as `DummyTalonEnv` plus `headless`, so
  `scripts/train_prelim.py` (Task 5) can construct either with a near-
  identical call site.

- [ ] **Step 1: Write the structural test first (fails on any 3.12 venv without isaacsim)**

```python
# tests/test_isaac_lab_env.py
"""Structural contract test for IsaacLabTalonEnv — skips entirely on a venv
without Isaac Sim installed (e.g. the repo's default 3.12 .venv). The real
proof this env works is the manual GPU smoke run documented in README, not
this test — see docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md.
"""

import pytest

pytest.importorskip("isaacsim")

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.envs.isaac_lab_env import IsaacLabTalonEnv


def test_isaac_lab_env_implements_base_contract():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    env = IsaacLabTalonEnv(obs_cfg, action_cfg, horizon=10, headless=True)

    assert isinstance(env, BaseTalonEnv)
    assert env.num_envs == 1
    assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
    assert env.action_dim == action_cfg.dim

    transition = env.reset()
    assert transition["obs"].shape == (env.obs_dim,)
    for key in (
        "v_actual", "v_command", "obstacle_dist", "joint_torque", "joint_vel",
        "foot_contact_force", "action", "prev_action", "joint_acc",
    ):
        assert key in transition

    import numpy as np
    action = np.zeros(env.action_dim, dtype=np.float32)
    transition, done = env.step(action)
    assert transition["obs"].shape == (env.obs_dim,)
    assert isinstance(done, bool)
```

- [ ] **Step 2: Run it on the 3.12 `.venv` to confirm it skips cleanly**

Run:
```bash
cd "/home/xero/Master's Degree/Thesis/talon-rl"
.venv/bin/pytest tests/test_isaac_lab_env.py -v
```
Expected: `SKIPPED (could not import 'isaacsim')` — proves the no-GPU dev
path stays unaffected before any real implementation exists.

- [ ] **Step 3: Implement `IsaacLabTalonEnv`**

```python
# talon_rl/envs/isaac_lab_env.py
"""Real (non-dummy) Isaac Lab environment for the Unitree A1.

Same reset()/step() contract as DummyTalonEnv — training/moppo.py does not
change to use this instead. See docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md
for the full design rationale (why a direct SimulationContext wrapper
instead of Isaac Lab's DirectRLEnv, why num_envs=1, PD-gain provenance).

The `clearance` reward term's obstacle_dist is still a scripted placeholder
here (no Exteroception Module in this prelim) — mirrors what DummyTalonEnv
does, per the design doc's explicit note not to silently drop the term.
"""

from __future__ import annotations

import numpy as np

from ..config import ActionSpaceCfg, ObservationSpaceCfg
from .base_env import BaseTalonEnv

# Filled in from Task 2's inspection output — replace with the actual
# printed values before running this for real.
_A1_ACTUATOR_GROUP = "legs"  # <- confirm against Task 2 Step 1's printed key
_A1_KP = 55.0  # RMA (Kumar et al. 2021) — see design doc Risks section
_A1_KD = 0.8


class IsaacLabTalonEnv(BaseTalonEnv):
    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        horizon: int = 200,
        dt: float = 0.02,
        headless: bool = True,
    ):
        # Imported here, not at module top, so `import talon_rl.envs.isaac_lab_env`
        # never fails on a machine without Isaac Sim installed (train_prelim.py's
        # --env dummy path must keep working everywhere).
        from isaacsim import SimulationApp

        self._app = SimulationApp({"headless": headless})

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab_assets import UNITREE_A1_CFG

        self.num_envs = 1
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self.horizon = horizon
        self.dt = dt

        self._sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())

        robot_cfg = UNITREE_A1_CFG.replace(prim_path="/World/Robot")
        robot_cfg.actuators[_A1_ACTUATOR_GROUP].stiffness = _A1_KP
        robot_cfg.actuators[_A1_ACTUATOR_GROUP].damping = _A1_KD
        self._robot = Articulation(robot_cfg)

        # Foot contact: 4 feet, per ObservationSpaceCfg.foot_contact_dim.
        # Body names follow A1's USD naming; confirm against Task 2's
        # robot.body_names if this regex doesn't match any body.
        self._contact_sensor = ContactSensor(
            ContactSensorCfg(prim_path="/World/Robot/.*_foot", history_length=1)
        )

        self._sim.reset()
        self._robot.reset()

        self.t = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.v_command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        # Scripted obstacle, same role as DummyTalonEnv's — no real
        # Exteroception signal in this prelim.
        self._obstacle_ahead = 5.0

    def reset(self) -> dict:
        self.t = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._robot.reset()
        self._sim.reset()
        return self._build_transition(action=np.zeros(self.action_dim, dtype=np.float32))

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # action_cfg.dim target joint angles -> Articulation applies through
        # its ImplicitActuator/DCMotor PD model using the Kp/Kd set above.
        joint_pos_target = torch_from_numpy(action)  # see helper below
        self._robot.set_joint_position_target(joint_pos_target)
        self._robot.write_data_to_sim()

        self._sim.step()
        self._robot.update(self.dt)
        self._contact_sensor.update(self.dt)

        transition = self._build_transition(action)
        self.prev_action = action
        self.t += 1

        pos_x = float(self._robot.data.root_state_w[0, 0].item())
        done = self.t >= self.horizon or pos_x >= self._obstacle_ahead
        return transition, done

    def _build_transition(self, action: np.ndarray) -> dict:
        import torch

        joint_pos = self._robot.data.joint_pos[0].cpu().numpy()
        joint_vel = self._robot.data.joint_vel[0].cpu().numpy()
        root_quat = self._robot.data.root_state_w[0, 3:7].cpu().numpy()  # (w, x, y, z)
        lin_vel = self._robot.data.root_state_w[0, 7:10].cpu().numpy()

        roll_pitch = _quat_to_roll_pitch(root_quat)

        contact_forces = self._contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
        foot_contact_force = torch.norm(contact_forces[0], dim=-1).cpu().numpy()
        foot_contact_binary = (foot_contact_force > 1.0).astype(np.float32)

        joint_acc = (action - self.prev_action) / self.dt
        joint_torque = self._robot.data.applied_torque[0].cpu().numpy()

        obs = np.concatenate(
            [
                joint_pos.astype(np.float32),
                joint_vel.astype(np.float32),
                roll_pitch.astype(np.float32),
                foot_contact_binary[: self.obs_cfg.foot_contact_dim],
                self.prev_action[: self.obs_cfg.prev_action_dim],
                self.v_command,
            ]
        )

        return {
            "obs": obs,
            "v_actual": np.array([lin_vel[0], lin_vel[1], 0.0], dtype=np.float32),
            "v_command": self.v_command,
            "obstacle_dist": max(0.0, self._obstacle_ahead - float(self._robot.data.root_state_w[0, 0].item())),
            "joint_torque": joint_torque.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "joint_acc": joint_acc.astype(np.float32),
            "foot_contact_force": foot_contact_force.astype(np.float32),
            "action": action,
            "prev_action": self.prev_action,
        }

    def close(self) -> None:
        self._app.close()


def torch_from_numpy(arr: np.ndarray):
    import torch

    return torch.from_numpy(arr).unsqueeze(0)  # (1, action_dim) — batch dim for num_envs=1


def _quat_to_roll_pitch(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> (roll, pitch) in radians, standard aerospace convention."""
    w, x, y, z = quat_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return np.array([roll, pitch], dtype=np.float32)
```

**Known gaps to note in a code comment where marked, not silently fixed
here:**
- `_A1_ACTUATOR_GROUP` and the `.*_foot` body regex are written from Task
  2's expected findings — if Task 2's actual printed output differs
  (different dict key, different body naming), update these two constants
  and the regex before running Task 4's manual smoke test. This is a known
  follow-up, not a design fork — the structural test (Step 1) doesn't
  exercise a live sim, so it can't catch a wrong key/regex; only the manual
  GPU smoke run does.

- [ ] **Step 4: Run the structural test on the Isaac Lab venv**

Run:
```bash
source ~/isaac-lab-env/bin/activate
cd "/home/xero/Master's Degree/Thesis/talon-rl"
pytest tests/test_isaac_lab_env.py -v
```
Expected: PASS. If it fails with an attribute/key error from the constants
noted above, fix them against Task 2's actual recorded output and rerun —
do not guess a second time, re-run Task 2's inspection script if unsure.

- [ ] **Step 5: Commit**

```bash
git add talon_rl/envs/isaac_lab_env.py tests/test_isaac_lab_env.py
git commit -m "feat: add IsaacLabTalonEnv implementing BaseTalonEnv contract"
```

---

### Task 4: Wire `--env {dummy,isaac_lab}` into `scripts/train_prelim.py`

**Files:**
- Modify: `scripts/train_prelim.py`

**Interfaces:**
- Consumes: `DummyTalonEnv(obs_cfg, action_cfg, horizon, seed)` (existing);
  `IsaacLabTalonEnv(obs_cfg, action_cfg, horizon, headless)` (Task 3).

- [ ] **Step 1: Add the flag and lazy branch**

```python
# scripts/train_prelim.py — replace the body of main()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    args = parser.parse_args()

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    if args.env == "dummy":
        env = DummyTalonEnv(obs_cfg, action_cfg, horizon=200, seed=args.seed)
    else:
        # Imported lazily so `--env dummy` keeps working on machines without
        # Isaac Sim installed (this repo's default 3.12 .venv included).
        from talon_rl.envs.isaac_lab_env import IsaacLabTalonEnv

        env = IsaacLabTalonEnv(obs_cfg, action_cfg, horizon=200, headless=True)

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
        env.close()
```

- [ ] **Step 2: Confirm the dummy path still works unchanged**

Run:
```bash
cd "/home/xero/Master's Degree/Thesis/talon-rl"
.venv/bin/python scripts/train_prelim.py --updates 3
```
Expected: identical output to before this change (3 update lines, no
errors) — proves the default `--env dummy` path (and machines without Isaac
Sim) are unaffected.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_prelim.py
git commit -m "feat: add --env isaac_lab flag to train_prelim.py"
```

---

### Task 5: Manual GPU smoke test and README update

**Files:**
- Modify: `README.md`

**Interfaces:** none (this task's deliverable is a verified manual run plus
a documented command, not new code).

- [ ] **Step 1: Run the real smoke test**

```bash
source ~/isaac-lab-env/bin/activate
cd "/home/xero/Master's Degree/Thesis/talon-rl"
python scripts/train_prelim.py --env isaac_lab --updates 5
```
Expected: 5 update lines print with finite numbers, same shape as the dummy
env's output, no crash. This is the actual proof this milestone's goal is
met — per the spec's Non-goals, do **not** interpret the printed reward
values as a locomotion result of any kind, only as "the loop completed."

- [ ] **Step 2: Document the manual step in README**

Add to `README.md`'s "Running it" section (after the existing
`pytest tests/` / `python scripts/train_prelim.py --updates 50` lines):

```markdown
### Isaac Lab smoke test (requires the separate `~/isaac-lab-env` venv, GPU machine only)

```bash
source ~/isaac-lab-env/bin/activate
python scripts/train_prelim.py --env isaac_lab --updates 5
```

Proves the pipeline runs against a real Isaac Lab environment instead of
`DummyTalonEnv` — same "doesn't crash" bar, not a locomotion result. See
`docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md` for setup
details and known limitations (VRAM ceiling, actuator gain provenance).
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the Isaac Lab smoke-test manual step"
```

---

## Plan self-review notes

- **Spec coverage:** §1 (env setup) -> Task 1. §2 (`IsaacLabTalonEnv`) ->
  Tasks 2-3. §3 (wiring) -> Task 4. §4 (testing strategy) -> Tasks 3 (structural
  test) and 5 (manual smoke run). Risks section's $K_p$/$K_d$ finding -> baked
  into Task 3's `_A1_KP`/`_A1_KD` constants with the actuator-group-name
  uncertainty resolved by Task 2's inspection step rather than a guess.
- **Known deliberate gap:** the exact `_A1_ACTUATOR_GROUP` dict key and
  `.*_foot` body regex in Task 3's code are written as the *expected* values
  from research, not confirmed against a real install (none exists yet) —
  Task 2 exists specifically to confirm or correct them before Task 3's
  Step 4 runs for real. This is flagged explicitly in Task 3, not hidden.
