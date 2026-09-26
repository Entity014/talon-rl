# talon_rl/tasks/locomotion/a1_env/a1_env.py
"""IsaacLabTalonEnv(ManagerBasedRLEnv) — the class registered as
Isaac-Talon-A1-v0. Scene/observations/actions/terminations/events all come
from IsaacLabTalonEnvCfg's manager configs (a1_env_cfg.py) — this class
only adds: (1) the v_command/obstacle_ahead scripted buffers
mdp/observations.py and mdp/terminations.py read, and (2) step()/reset()
overrides that return this repo's TalonEnv-shaped
(transition_dict, done_array) instead of gym.Env's raw tuple, since nothing
outside this repo needs generic gym.Env compliance from this class
(train_prelim.py drives it directly via the TalonEnv contract) — see
docs/superpowers/specs/2026-09-14-vectorized-isaac-lab-env-design.md.
"""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from talon_rl.rewards import directed_progress
from talon_rl.curricula.command_schedule import G1CommandSchedule

from .a1_env_cfg import IsaacLabTalonEnvCfg
from .mdp.observations import roll_pitch as _roll_pitch
from .mdp.observations import yaw as _yaw


class IsaacLabTalonEnv(ManagerBasedRLEnv):
    cfg: IsaacLabTalonEnvCfg

    def __init__(self, cfg: IsaacLabTalonEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # NOTE deviation from the brief's literal code (verified against the
        # installed isaaclab 0.48.0 source, not guessed): ManagerBasedEnv.num_envs
        # is a read-only @property (`return self.scene.num_envs`, no setter) --
        # `self.num_envs = self.cfg.scene.num_envs` here raised
        # AttributeError: can't set attribute. The inherited property already
        # returns the same value, so the assignment is simply dropped.
        # v_command_buf/obstacle_ahead_buf construction moved to the
        # load_managers() override below -- see its docstring for why.
        self.obs_dim = self.cfg.obs_dim
        self.action_dim = self.cfg.action_dim
        if not hasattr(self, "_g1_schedule"):
            self._g1_schedule = None

    def load_managers(self) -> None:
        """Overridden (deviation from the brief's literal code, verified against
        isaaclab.envs.manager_based_env.ManagerBasedEnv.__init__/.load_managers,
        not guessed): ManagerBasedEnv.__init__ builds self.scene, THEN calls
        self.load_managers() -- which synchronously constructs ObservationManager,
        invoking every term function once (incl. mdp.v_command /
        mdp.obstacle_reached) to infer shapes -- all before control ever returns
        to IsaacLabTalonEnv.__init__'s own body. So v_command_buf/obstacle_ahead_buf
        must exist by the time load_managers() runs, not after super().__init__()
        returns. self.scene already exists at this point (scene creation happens
        before ManagerBasedEnv.__init__ calls self.load_managers()), so
        self.device/self.num_envs (both properties reading self.scene) are valid
        here.
        """
        self.v_command_buf = torch.tensor([0.5, 0.0, 0.0], device=self.device).expand(self.num_envs, 3).contiguous()
        if self.cfg.g1_command_exposure:
            self._g1_schedule = G1CommandSchedule(
                self.num_envs, np.random.default_rng(self.cfg.seed),
            )
        self.obstacle_ahead_buf = torch.full((self.num_envs,), 5.0, device=self.device)
        # Consecutive-success counter for mdp.terrain_levels_vel's promotion
        # dwell requirement — see that function's docstring.
        self.terrain_promote_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        super().load_managers()
        # impact_reward's foot-slip sub-term (2026-09-18) needs per-foot
        # linear velocity indexed IN THE SAME ORDER as foot_contact_force
        # (contact_sensor.data.net_forces_w), so the two can be multiplied
        # elementwise in _reward_fields. ContactSensor resolves its own
        # body order by PhysX rigid-body-view glob discovery, which is not
        # guaranteed to match Articulation.find_bodies's body-index order —
        # so look up body ids BY THE SENSOR'S OWN NAMES (preserve_order=True)
        # rather than re-matching ".*_foot" independently and assuming the
        # two orderings agree.
        contact_sensor = self.scene.sensors["contact_sensor"]
        self._foot_body_ids, _ = self.scene["robot"].find_bodies(contact_sensor.body_names, preserve_order=True)
        # progress_reward's feet-air-time sub-term (2026-09-18, legged_gym/
        # Rudin et al. 2022 -- see reward.py's docstring for the formula and
        # why it was added). Per-foot seconds since last touchdown; reset in
        # _reset_idx below, advanced/consumed in _reward_fields.
        n_feet = len(self._foot_body_ids)
        self._foot_air_time = torch.zeros(self.num_envs, n_feet, device=self.device)
        self._foot_last_contact = torch.zeros(self.num_envs, n_feet, dtype=torch.bool, device=self.device)
        self._foot_air_time_reward = torch.zeros(self.num_envs, device=self.device)
        self._foot_air_time_last_step = -1  # see _compute_foot_air_time_reward's docstring
        # balance_reward's hip_activation/hip_sym sub-terms (2026-09-20,
        # Experiment 2A.4-5) -- L = {FL, RL}, R = {FR, RR}, same
        # front/rear-mirrors-front/rear grouping used throughout that
        # session's diagnostic scripts (gait_joint_trace.py etc.), found
        # via UNITREE_A1_CFG's own default standing pose (FL_hip=+0.1,
        # FR_hip=-0.1, RL_hip=+0.1, RR_hip=-0.1 -- left/right hip angles
        # are sign-mirrored about 0 at the symmetric stance, not equal).
        self._hip_joint_ids_L, _ = self.scene["robot"].find_joints(["FL_hip_joint", "RL_hip_joint"], preserve_order=True)
        self._hip_joint_ids_R, _ = self.scene["robot"].find_joints(["FR_hip_joint", "RR_hip_joint"], preserve_order=True)
        # progress_reward's directed-progress sub-term (R1, 2026-09-20, see
        # artifacts/r1_freeze/FREEZE.md and talon_rl/directed_progress.py
        # for the frozen formula/pure logic this just supplies inputs to).
        # Origin values start at zero/unset -- harmless, since
        # _dp_just_reset starts True for every lane, so the first real
        # _reward_fields() call re-anchors each lane's window at whatever
        # its actual post-reset spawn pose is, not this placeholder.
        self._dp_origin_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._dp_origin_heading = torch.zeros(self.num_envs, device=self.device)
        self._dp_origin_command_x = self.v_command_buf[:, 0].clone()
        self._dp_just_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, **kwargs) -> dict:
        obs_dict, _extras = super().reset(**kwargs)
        if self._g1_schedule is not None:
            self.v_command_buf[:] = torch.from_numpy(self._g1_schedule.command).to(self.device)
        return self._transition(obs_dict)

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        action_t = torch.from_numpy(action).to(self.device)
        # ManagerBasedRLEnv resets done lanes inside super().step(). Its
        # _reset_idx hook below snapshots reward inputs immediately before
        # that reset, while physics/action state still belongs to the action
        # just applied.
        self._capture_terminal_reward = True
        self._terminal_fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-term termination breakdown (2026-09-18) -- terminal_fall above
        # is actually "real termination, fall OR obstacle_reached" (anything
        # not time_out), which is what GAE needs but conflates the two when
        # diagnosing WHY mean_episode_length is stuck: was training-loop
        # investigation this session guessing from a single aggregate
        # number every time. These three are mutually exclusive per env per
        # step (Isaac Lab's TerminationManager only fires one cause per
        # reset), captured the same way terminal_fall already is.
        self._term_time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_obstacle_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            obs_dict, _reward_buf, terminated, truncated, _extras = super().step(action_t)
        finally:
            self._capture_terminal_reward = False
        if self._g1_schedule is not None:
            event = self._g1_schedule.before_observation()
            self.v_command_buf[:] = torch.from_numpy(event.command).to(self.device)
            if event.progress_reset.any():
                self._dp_just_reset[torch.from_numpy(np.flatnonzero(event.progress_reset)).to(self.device)] = True
            self._g1_last_event = event
            self._g1_last_dp_reset = event.progress_reset.copy()
        transition = self._transition(obs_dict)
        done = (terminated | truncated).cpu().numpy()
        transition["terminal_fall"] = self._terminal_fall.cpu().numpy()
        transition["term_time_out"] = self._term_time_out.cpu().numpy()
        transition["term_obstacle_reached"] = self._term_obstacle_reached.cpu().numpy()
        transition["term_base_contact"] = self._term_base_contact.cpu().numpy()
        if done.any():
            reward_transition = {key: value.copy() for key, value in transition.items()}
            for key, value in self._terminal_reward_fields.items():
                reward_transition[key][done] = value[done]
            transition["reward_transition"] = reward_transition
        return transition, done

    def _reset_idx(self, env_ids) -> None:
        """Snapshot terminal reward fields before Isaac Lab overwrites a done lane.

        The parent calls this hook after termination/reward bookkeeping but
        before post-reset observations are computed. During an explicit reset
        (including construction), capture is disabled and this remains the
        normal Isaac Lab reset path.
        """
        if getattr(self, "_capture_terminal_reward", False):
            self._terminal_reward_fields = self._reward_fields()
            self._terminal_fall[env_ids] = self.termination_manager.terminated[env_ids]
            def _term_or_false(name: str) -> torch.Tensor:
                try:
                    return self.termination_manager.get_term(name)
                except KeyError:
                    # Evaluation may deliberately disable a termination
                    # (e.g. the training-only 5 m obstacle sentinel). Keep
                    # the stable transition schema and report it as false.
                    return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            self._term_time_out[env_ids] = _term_or_false("time_out")[env_ids]
            self._term_obstacle_reached[env_ids] = _term_or_false("obstacle_reached")[env_ids]
            self._term_base_contact[env_ids] = _term_or_false("base_contact")[env_ids]
            # Clear the just-terminated lanes' foot-air-time state AFTER the
            # snapshot above already read it (that snapshot needs this
            # episode's real final swing-phase timing) -- a fresh episode
            # shouldn't inherit stale air-time from the previous one.
            self._foot_air_time[env_ids] = 0.0
            self._foot_last_contact[env_ids] = False
        # directed-progress window (R1, 2026-09-20) -- mark these lanes so
        # the NEXT _reward_fields() call (which sees the post-reset spawn
        # pose, set by super()._reset_idx below) re-anchors their window
        # there instead of carrying over a stale pre-reset origin.
        # Unconditional (both branches, not just the capture-enabled one):
        # an explicit env.reset() (capture disabled, e.g. construction)
        # still needs this the same way.
        self._dp_just_reset[env_ids] = True
        if self._g1_schedule is not None:
            done = np.zeros(self.num_envs, dtype=bool)
            ids = env_ids.cpu().numpy() if hasattr(env_ids, "cpu") else np.asarray(env_ids)
            done[ids] = True
            self._g1_schedule.reset_lanes(done)
            self.v_command_buf[env_ids] = torch.from_numpy(self._g1_schedule.command[ids]).to(self.device)
        super()._reset_idx(env_ids)

    def _compute_foot_air_time_reward(self) -> torch.Tensor:
        """progress_reward's feet-air-time sub-term (2026-09-18, legged_gym/
        Rudin et al. 2022 -- see reward.py's progress_reward docstring for
        the formula and why it was added). Mutates self._foot_air_time,
        so it must run EXACTLY ONCE per real physics step -- but
        _reward_fields() is called TWICE within a single step() whenever
        any lane terminates this step (once from _reset_idx's terminal
        snapshot, before that lane's physics reset; once more from the
        regular post-step _transition() call -- see step()'s own
        docstring). With ~90%+ of steps having at least one of 4096 lanes
        terminate once training progresses, mutating on both calls would
        double-increment/double-consume air time for the WHOLE batch on
        nearly every step, not just the terminating lanes. Guarded here by
        self.common_step_counter, Isaac Lab's own once-per-physics-step
        counter: a second call within the same step just returns the
        already-computed cached value instead of re-mutating."""
        if self._foot_air_time_last_step != self.common_step_counter:
            contact_force = torch.norm(self.scene.sensors["contact_sensor"].data.net_forces_w, dim=-1)
            contact = contact_force > 1.0
            # OR with the previous step's contact (before it's overwritten
            # below) -- a two-frame filter against a single-frame contact-
            # sensor false negative, same as legged_gym's own
            # implementation of this term.
            contact_filt = contact | self._foot_last_contact
            first_contact = (self._foot_air_time > 0.0) & contact_filt
            self._foot_air_time = self._foot_air_time + self.step_dt
            self._foot_air_time_reward = torch.sum(
                (self._foot_air_time - 0.5) * first_contact.float(), dim=-1
            )
            self._foot_air_time = self._foot_air_time * (~contact_filt).float()
            self._foot_last_contact = contact
            self._foot_air_time_last_step = self.common_step_counter
        return self._foot_air_time_reward

    def _transition(self, obs_dict: dict) -> dict:
        transition = self._reward_fields()
        transition.update({
            "obs": obs_dict["policy"].cpu().numpy().astype(np.float32),
            "extrinsics": obs_dict["privileged"].cpu().numpy().astype(np.float32),
        })
        return transition

    def _reward_fields(self) -> dict:
        """Reward inputs for the current physical frame, before any reset."""
        robot = self.scene["robot"]
        prev_action = self.action_manager.prev_action.cpu().numpy()
        action = self.action_manager.action.cpu().numpy()
        return {
            # Evaluation-frame pose. Kept in the transition so terminal
            # snapshots preserve the pose that caused termination rather
            # than the auto-reset pose returned by Isaac Lab afterward.
            "root_pos_w": robot.data.root_pos_w.cpu().numpy(),
            "root_quat_w": robot.data.root_quat_w.cpu().numpy(),
            "root_lin_vel_w": robot.data.root_lin_vel_w.cpu().numpy(),
            # v_command is (v_x, v_y, omega_z) (config.py's command_dim comment) —
            # root_lin_vel_b alone is (v_x, v_y, v_z), so its 3rd column was being
            # compared against a yaw-rate target instead of the robot's actual yaw
            # rate. Swap in root_ang_vel_b's z-component so progress_reward's
            # exp-kernel tracks what v_command actually specifies.
            "v_actual": torch.cat(
                [robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=-1
            ).cpu().numpy(),
            "v_command": self.v_command_buf.cpu().numpy(),
            # Vertical (body-frame z) linear velocity -- balance_reward's
            # z-acceleration sub-term (2026-09-18, see that function's
            # docstring), penalizing bouncing/vertical bobbing. Despite the
            # reference term's name ("Z Acceleration"), its own formula
            # uses velocity (-||v_z||^2), not d(v_z)/dt -- matched here, not
            # the literal name.
            "v_z": robot.data.root_lin_vel_b[:, 2].cpu().numpy(),
            # Height above this lane's own spawn origin (2026-09-18) --
            # balance_reward's crouch sub-term. v_z alone only penalizes
            # vertical MOTION (bouncing) -- a lane that crouches down low
            # and then holds perfectly still gets v_z=0 (no penalty) forever,
            # exactly the "stand nearly still" local optimum a forced-
            # command physics probe found this same day (6.7% tracking
            # ratio, 81% action saturation, unchanged by the other grouped
            # sub-penalties). Same env_origins-subtraction pattern as
            # obstacle_dist below, since root_pos_w is world-frame and each
            # lane spawns on a different terrain cell. NOTE: this is height
            # above SPAWN, not height above local ground directly under the
            # robot right now -- on sloped/uneven terrain within a cell
            # these can diverge; a real height-scanner sensor would be more
            # accurate but doesn't exist in this prelim (see clearance's own
            # "no Exteroception Module yet" note above _TERM_FUNCS in
            # reward.py). Good enough to catch a lane crouched near its own
            # spawn point, which is what the probe actually showed.
            "height": (robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]).cpu().numpy(),
            "roll_pitch": _roll_pitch(self).cpu().numpy(),  # balance_reward's dense anti-fall signal
            # balance_reward's hip_activation/hip_sym sub-terms (2026-09-20)
            # -- mean joint angular velocity / position over each side's hip
            # pair, real physics (robot.data.joint_vel/joint_pos), not a
            # commanded-target proxy. See load_managers's own comment on
            # _hip_joint_ids_L/_R for the L/R grouping and why it's a mirror
            # (not equality) relationship.
            "hip_qdot_L": robot.data.joint_vel[:, self._hip_joint_ids_L].mean(dim=-1).cpu().numpy(),
            "hip_qdot_R": robot.data.joint_vel[:, self._hip_joint_ids_R].mean(dim=-1).cpu().numpy(),
            "hip_q_L": robot.data.joint_pos[:, self._hip_joint_ids_L].mean(dim=-1).cpu().numpy(),
            "hip_q_R": robot.data.joint_pos[:, self._hip_joint_ids_R].mean(dim=-1).cpu().numpy(),
            # root_ang_vel_b's x/y components (2026-09-19) -- real physics
            # angular velocity, not a finite-difference approximation of
            # roll_pitch. Added after a 438-fall-event calibration
            # (prefall_window_analysis.py) found |pitch_rate| separates
            # pre-fall from normal-walking states ~3.2-3.6x (p50/p90),
            # substantially more than |pitch| itself (~2x) -- balance_reward's
            # own tilt_rate_coef term.
            "roll_pitch_rate": robot.data.root_ang_vel_b[:, 0:2].cpu().numpy(),
            # root_pos_w is world-frame; subtract env_origins.x so this is
            # distance-to-obstacle from the env's own spawn, not absolute
            # world x (same fix as mdp/terminations.py's obstacle_reached —
            # the terrain grid spreads env_origins.x well past 5.0).
            "obstacle_dist": np.maximum(
                0.0,
                (self.obstacle_ahead_buf - (robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0])).cpu().numpy(),
            ),
            "joint_torque": robot.data.applied_torque.cpu().numpy(),
            "joint_vel": robot.data.joint_vel.cpu().numpy(),
            "joint_acc": (action - prev_action) / self.step_dt,
            "foot_contact_force": torch.norm(
                self.scene.sensors["contact_sensor"].data.net_forces_w, dim=-1
            ).cpu().numpy(),
            # impact_reward's foot-slip sub-term (2026-09-18, see its
            # docstring) -- world-frame per-foot linear velocity, (N, 4, 3),
            # in the same foot order as foot_contact_force above (see the
            # self._foot_body_ids comment in load_managers for why it's
            # looked up via the sensor's own body_names, not re-matched).
            "foot_vel": robot.data.body_lin_vel_w[:, self._foot_body_ids, :].cpu().numpy(),
            # progress_reward's feet-air-time sub-term -- see
            # _compute_foot_air_time_reward's docstring for the
            # double-call-per-step guard this relies on.
            "foot_air_time_reward": self._compute_foot_air_time_reward().cpu().numpy(),
            # impact_reward's undesired-contact sub-term (2026-09-19, see
            # a1_env_cfg.py's undesired_contact_sensor comment) -- COUNT of
            # calf (shin) bodies touching anything, no force threshold
            # (unlike foot_contact_force's peak-impact logic above, which
            # is built for transient landing shocks and would miss a
            # sustained LOW-force drag entirely). A calf should never
            # register any contact force during normal operation, so any
            # nonzero reading here is undesired regardless of magnitude.
            "undesired_contact_count": (
                torch.norm(self.scene.sensors["undesired_contact_sensor"].data.net_forces_w, dim=-1) > 1.0
            ).sum(dim=-1).float().cpu().numpy(),
            "undesired_contact_force": torch.norm(
                self.scene.sensors["undesired_contact_sensor"].data.net_forces_w, dim=-1
            ).cpu().numpy(),
            "action": action,
            "prev_action": prev_action,
            # progress_reward's directed-progress sub-term (R1, 2026-09-20,
            # see artifacts/r1_freeze/FREEZE.md). Called on every
            # _reward_fields() invocation -- including the terminal-frame
            # snapshot call from _reset_idx above, BEFORE physics reset --
            # unlike _compute_foot_air_time_reward this needs no
            # once-per-physics-step guard: it's a pure function of current
            # position/heading/command, not an accumulating counter, so a
            # non-resetting lane recomputing the identical value from the
            # identical (unchanged-between-calls) pose is harmless, and a
            # resetting lane correctly gets scored once against its OLD
            # window (this call, pre-reset pose, _dp_just_reset still False
            # for it) and once against its NEW window (the next
            # _reward_fields() call, post-reset pose, _dp_just_reset now
            # True -- see _reset_idx's ordering).
            "directed_progress": self._compute_directed_progress(robot),
        }

    def _compute_directed_progress(self, robot) -> np.ndarray:
        state = directed_progress.DirectedProgressState(
            origin_xy=self._dp_origin_xy.cpu().numpy(),
            origin_heading=self._dp_origin_heading.cpu().numpy(),
            origin_command_x=self._dp_origin_command_x.cpu().numpy(),
        )
        value, new_state = directed_progress.update(
            state,
            root_xy=robot.data.root_pos_w[:, :2].cpu().numpy(),
            heading=_yaw(self).cpu().numpy(),
            command_x=self.v_command_buf[:, 0].cpu().numpy(),
            reset_mask=self._dp_just_reset.cpu().numpy(),
        )
        self._dp_origin_xy = torch.from_numpy(new_state.origin_xy).to(self.device, dtype=torch.float32)
        self._dp_origin_heading = torch.from_numpy(new_state.origin_heading).to(self.device, dtype=torch.float32)
        self._dp_origin_command_x = torch.from_numpy(new_state.origin_command_x).to(self.device, dtype=torch.float32)
        self._dp_just_reset[:] = False
        return value
