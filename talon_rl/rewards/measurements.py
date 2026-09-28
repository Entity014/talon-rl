"""Raw measurement library (not rewards).

Physical measurements that the stock reward library does not cover, logged
for feature selection. They have no reward weight and enter no objective or
training reward. A zero-weight IsaacLab reward term is not computed at all
(RewardManager skips weight 0.0), so these are computed here from the
simulator state instead. Costs are unweighted and non-negative (higher =
more of the quantity). Roles are initial screening roles, not decisions.
See talon_rl/rewards/README.md.
"""
from __future__ import annotations

import torch

# name: (tier, initial role, meaning)
MEASUREMENT_LIBRARY = {
    "base_lin_acc_z_l2": (1, "objective / constraint candidate", "(Δ v_z world / dt)^2, vertical acceleration"),
    "body_height_osc_l2": (1, "constraint / V candidate", "(z - EMA(z))^2, vertical oscillation about a 0.5 s moving mean (not posture)"),
    "foot_slip": (1, "constraint candidate", "sum over feet in contact of |v_xy| (IsaacLab feet_slide form)"),
    "foot_impact_l2": (1, "constraint candidate", "sum over feet of |ΔF_contact|^2 between policy steps"),
    "dof_vel_l2": (2, "regularizer candidate", "sum of joint velocity^2"),
    "action_magnitude_l2": (2, "regularizer candidate", "sum of action^2 (effort / saturation, not rate)"),
    "joint_power_abs": (2, "regularizer candidate", "sum |tau * qdot|, mechanical power"),
    "root_z": (0, "raw channel", "world root height (target for vertical excursion)"),
    "v_z_world": (0, "raw channel", "world-frame vertical velocity"),
}
NAMES = tuple(MEASUREMENT_LIBRARY)
CONTACT_N = 1.0  # N, same threshold as IsaacLab feet_slide
HEIGHT_EMA_TAU = 0.5  # s. Oscillation, not a fixed nominal height: default_root_state z is the spawn height, and an
                      # offset from a fixed height is posture (O-like), not vertical dynamics.


def ema_update(mean, x, dt, tau=HEIGHT_EMA_TAU):
    return mean + (dt / tau) * (x - mean)


def vertical_acc_sq(vz, vz_prev, dt):
    return ((vz - vz_prev) / dt) ** 2


def impact_sq(force, force_prev):
    """force [N, feet, 3] -> [N]"""
    return ((force - force_prev) ** 2).sum(-1).sum(-1)


def foot_slip(vel_xy, force):
    """vel_xy [N, feet, 2], force [N, feet, 3] -> [N]"""
    return (vel_xy.norm(dim=-1) * (force.norm(dim=-1) > CONTACT_N)).sum(-1)


def joint_power_abs(tau, qd):
    return (tau * qd).abs().sum(-1)


class MeasurementLibrary:
    """Stateful per-step logger. Call reset() after every reset or state restore."""

    def __init__(self, env):
        u = env.unwrapped
        self.robot, self.sensor, self.dt = u.scene["robot"], u.scene["contact_forces"], u.step_dt
        self.feet_sensor = self.sensor.find_bodies(".*_foot")[0]
        self.feet_body = self.robot.find_bodies(".*_foot")[0]

    def reset(self):
        self.vz_prev = self.robot.data.root_lin_vel_w[:, 2].clone()
        self.f_prev = self.sensor.data.net_forces_w[:, self.feet_sensor].clone()
        self.z_mean = self.robot.data.root_pos_w[:, 2].clone()

    def __call__(self, action) -> torch.Tensor:
        d = self.robot.data
        vz = d.root_lin_vel_w[:, 2]; f = self.sensor.data.net_forces_w[:, self.feet_sensor]
        z = d.root_pos_w[:, 2]; self.z_mean = ema_update(self.z_mean, z, self.dt)
        out = torch.stack([
            vertical_acc_sq(vz, self.vz_prev, self.dt),
            (z - self.z_mean) ** 2,
            foot_slip(d.body_lin_vel_w[:, self.feet_body, :2], f),
            impact_sq(f, self.f_prev),
            (d.joint_vel ** 2).sum(-1),
            (action ** 2).sum(-1),
            joint_power_abs(d.applied_torque, d.joint_vel),
            d.root_pos_w[:, 2],
            vz,
        ], -1)
        self.vz_prev, self.f_prev = vz.clone(), f.clone()
        return out
