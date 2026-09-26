"""B0 scalar-reward composition wrapper around the unchanged A1 environment."""
from __future__ import annotations
import numpy as np
import torch
from ..rewards.baselines import b0_reward


class B0TalonEnv:
    command = (0.5, 0.0, 0.0)
    def __init__(self, env):
        self.env = env
        self.num_envs, self.device = env.num_envs, env.device
        # Effective nominal root height, not a hard-coded constant.
        self.z_nominal = float(env.scene["robot"].data.default_root_state[0, 2].item())

    def _command(self):
        self.env.v_command_buf[:] = torch.tensor(self.command, device=self.device)

    def _scalar_transition(self, transition, done):
        # Isaac Lab auto-resets completed lanes before returning the next
        # observation. Reward inputs consequently come from
        # ``reward_transition`` (the pre-reset snapshot), while termination
        # reasons deliberately remain on the outer transition. Never infer a
        # fall from post-reset fields: that drops a real base-contact event or
        # charges it to the following episode.
        fields = transition.get("reward_transition", transition)
        done = np.asarray(done, dtype=bool)
        base_contact = np.asarray(
            transition.get("term_base_contact", fields.get("term_base_contact", np.zeros(self.num_envs, bool))),
            dtype=bool,
        )
        fall = base_contact & done
        reward = b0_reward(fields["v_actual"][:, 0], fields["roll_pitch"][:, 0],
                           fields["roll_pitch"][:, 1], fields["height"], self.z_nominal, fall)
        if reward.shape != (self.num_envs,) or not np.isfinite(reward).all():
            raise RuntimeError("B0 scalar reward must be finite shape (N,)")
        # Publish terminal metadata with the snapshot so callers can audit
        # exactly which action received a terminal penalty.
        audit_fields = dict(fields)
        audit_fields["term_base_contact"] = base_contact.copy()
        audit_fields["terminal_fall"] = fall.copy()
        return {"obs":transition["obs"], "reward":reward.astype(np.float32), "done":done,
                "terminal_fall":fall, "term_base_contact":base_contact,
                "fields":audit_fields,
                "command":np.broadcast_to(np.asarray(self.command,np.float32),(self.num_envs,3)).copy()}

    def reset(self):
        self.env.reset(); self._command()
        transition = self.env._transition(self.env.observation_manager.compute())
        return self._scalar_transition(transition, np.zeros(self.num_envs, bool))

    def step(self, actions):
        self._command()
        transition, done = self.env.step(actions)
        self._command()  # auto-reset lanes begin the next observation with B0 command.
        return self._scalar_transition(transition, done)
