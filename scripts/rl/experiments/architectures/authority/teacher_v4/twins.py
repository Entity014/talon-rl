"""Same-state branching on Isaac-Talon-A1-V4C-v0: snapshot every env, then restore the snapshot into the same envs before each branch.

Copying a state into a different env (a "twin") does not work: the twin sits
at another env origin, float32 contact geometry rounds differently there,
and legged contact amplifies it (joint velocity 0.84 rad/s apart after one
step, with every PhysX property equal). Restoring into the same env keeps
the coordinates identical. The snapshot is written before *both* branches,
so both start from a freshly written state (same solver warm-start).

Isaac Lab has no state snapshot, so the pieces the next steps depend on are
saved by hand: root pose and velocity, joint position and velocity, the
action manager's current and previous action and the joint-position term's
raw/processed actions, the velocity command term (command, heading target,
heading/standing flags, resampling timer, counter), the episode step
counter, and the observations. Masses/materials do not change after startup.
"""
from __future__ import annotations

import torch

CMD_FIELDS = ("vel_command_b", "heading_target", "is_heading_env", "is_standing_env", "time_left", "command_counter")


def snapshot(env, obs: dict) -> dict:
    u = env.unwrapped
    robot = u.scene["robot"]
    am = u.action_manager
    term = am.get_term("joint_pos")
    cmd = u.command_manager.get_term("base_velocity")
    return {"root": robot.data.root_state_w.clone(), "jpos": robot.data.joint_pos.clone(), "jvel": robot.data.joint_vel.clone(),
            "action": am._action.clone(), "prev_action": am._prev_action.clone(),
            "raw": term._raw_actions.clone(), "processed": term._processed_actions.clone(),
            "cmd": {k: getattr(cmd, k).clone() for k in CMD_FIELDS},
            "episode_length": u.episode_length_buf.clone(), "obs": {k: v.clone() for k, v in obs.items()}}


SETTLE = 4


def _write_body(robot, snap):
    robot.write_root_pose_to_sim(snap["root"][:, :7].clone())
    robot.write_root_velocity_to_sim(snap["root"][:, 7:13].clone())
    robot.write_joint_state_to_sim(snap["jpos"].clone(), snap["jvel"].clone())


def restore(env, snap: dict, settle: int = SETTLE) -> dict:
    """PhysX keeps a contact/warm-start cache that a state write does not
    clear, so each branch otherwise starts with the cache the previous branch
    left. Stepping physics `settle` times while re-writing the snapshot makes
    that cache come from the snapshot's own geometry in every branch."""
    u = env.unwrapped
    robot = u.scene["robot"]
    for _ in range(settle):
        _write_body(robot, snap)
        u.sim.step(render=False)
    _write_body(robot, snap)
    u.scene.update(dt=0.0)
    am = u.action_manager
    am._action[:] = snap["action"]; am._prev_action[:] = snap["prev_action"]
    term = am.get_term("joint_pos")
    term._raw_actions[:] = snap["raw"]; term._processed_actions[:] = snap["processed"]
    cmd = u.command_manager.get_term("base_velocity")
    for k in CMD_FIELDS:
        getattr(cmd, k)[:] = snap["cmd"][k]
    u.episode_length_buf[:] = snap["episode_length"]
    return {k: v.clone() for k, v in snap["obs"].items()}
