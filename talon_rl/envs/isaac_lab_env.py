"""Real (non-dummy) Isaac Lab environment for the Unitree A1.

Same reset()/step() contract as DummyTalonEnv — training/moppo.py does not
change to use this instead. See docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md
for the full design rationale (why a direct SimulationContext wrapper
instead of Isaac Lab's DirectRLEnv, why num_envs=1, PD-gain provenance).

The `clearance` reward term's obstacle_dist is still a scripted placeholder
here (no Exteroception Module in this prelim) — mirrors what DummyTalonEnv
does, per the design doc's explicit note not to silently drop the term.

Constants below (_A1_ACTUATOR_GROUP, joint order) are the *confirmed*
values from running the Task 2 inspection script against the real installed
UNITREE_A1_CFG (isaacsim 5.0.0.0 / isaaclab 0.48.0, 2026-09-14) — not a
guess. See the design doc's "Installed versions" section.
"""

from __future__ import annotations

import os

import numpy as np

from ..config import ActionSpaceCfg, ObservationSpaceCfg
from .base_env import BaseTalonEnv

# Confirmed against Task 2's inspection output — the shipped UNITREE_A1_CFG's
# single actuator group key is "base_legs", not "legs".
_A1_ACTUATOR_GROUP = "base_legs"
_A1_KP = 55.0  # RMA (Kumar et al. 2021) — see design doc Risks section
_A1_KD = 0.8

# ponytail: SimulationApp.close() hangs indefinitely on this isaacsim
# 5.0.0.0 install once a GPU-pipeline Articulation has been spawned/stepped
# (reproduced here and by Task 2's inspection script — a Kit/PhysX shutdown
# issue, not this repo's code). force-exit after this if graceful close
# doesn't return; nothing runs after close() anyway. Revisit if a later
# isaacsim patch fixes clean shutdown.
_CLOSE_TIMEOUT_S = 15.0


class IsaacLabTalonEnv(BaseTalonEnv):
    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        horizon: int = 200,
        dt: float = 0.02,
        headless: bool = True,
    ):
        # Isaac Sim's EULA prompt reads stdin — fatal in any non-interactive
        # run (pytest, train_prelim.py). Accepting here, not silently, is the
        # only way this constructor can succeed outside an interactive shell.
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

        # Imported here, not at module top, so `import talon_rl.envs.isaac_lab_env`
        # never fails on a machine without Isaac Sim installed (train_prelim.py's
        # --env dummy path must keep working everywhere).
        from isaacsim import SimulationApp

        self._app = SimulationApp({"headless": headless})

        import carb
        from isaacsim.storage.native import get_assets_root_path

        # isaaclab.utils.assets reads the Nucleus/CDN root straight from this
        # carb setting into a MODULE-LEVEL constant (computed once at import
        # time) — nothing in isaacsim 5.0.0.0 sets it on startup (confirmed
        # during Task 2: USD loads otherwise fail with a literal
        # "None/Isaac/..." path). Must run before the first `import
        # isaaclab.sim` anywhere below, or the constant bakes in as None.
        _root = get_assets_root_path()
        if _root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root (Nucleus/CDN unreachable)")
        carb.settings.get_settings().set("/persistent/isaac/asset_root/cloud", _root)

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

        # Foot contact: 4 feet, confirmed against Task 2's robot.body_names
        # (FL_foot, FR_foot, RL_foot, RR_foot).
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
        joint_pos_target = _torch_from_numpy(action)
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
        import threading

        watchdog = threading.Timer(_CLOSE_TIMEOUT_S, lambda: os._exit(0))
        watchdog.daemon = True
        watchdog.start()
        self._app.close()
        watchdog.cancel()


def _torch_from_numpy(arr: np.ndarray):
    import torch

    return torch.from_numpy(arr).unsqueeze(0)  # (1, action_dim) — batch dim for num_envs=1


def _quat_to_roll_pitch(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> (roll, pitch) in radians, standard aerospace convention."""
    w, x, y, z = quat_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return np.array([roll, pitch], dtype=np.float32)
