"""Ground-truth physics validation for play.py's --validate flag.

Formalizes a methodology repeated ad hoc via one-off scratchpad scripts
throughout 2026-09-18/19's debugging: training-time reward/episode-length
numbers repeatedly looked good while the underlying policy was actually
standing (nearly) still under a real command, or gaming a reward term via
some physically-implausible behavior a raw reward number can't reveal
(action saturation, torque limits, actual velocity tracking). This class
collects exactly those real physical quantities from the transition dict
during a play.py rollout and reports them, rather than trusting
Episode_Reward/* alone.

Only accumulates torque/saturation/roll_pitch/tracking stats for lanes
that haven't fallen yet (`alive` mask) — same convention as every
scratchpad probe this session, so a lane that fell early doesn't drag the
"how does a surviving lane behave" numbers down with post-fall garbage.
"""

from __future__ import annotations

import os

import numpy as np

A1_TORQUE_LIMIT_NM = 33.5  # Unitree A1 datasheet peak joint torque


class PhysicsValidator:
    def __init__(self, num_envs: int, action_clip: float, torque_limit_nm: float = A1_TORQUE_LIMIT_NM):
        self.num_envs = num_envs
        self.action_clip = action_clip
        self.torque_limit_nm = torque_limit_nm
        self.fall_step = np.full(num_envs, -1, dtype=np.int32)
        self._step = 0
        self._action_sat_frac: list[float] = []
        self._torque_max_per_step: list[float] = []
        self._torque_over_limit_frac: list[float] = []
        self._roll_pitch_max_per_step: list[float] = []
        self._v_actual_x: list[float] = []
        self._v_command_x: list[float] = []
        self._tracking_err: list[float] = []
        self._undesired_contact: list[np.ndarray] = []

    def record(self, transition: dict, action: np.ndarray, done: np.ndarray) -> None:
        alive = self.fall_step < 0
        if alive.any():
            sat = np.abs(action[alive]) >= 0.95 * self.action_clip
            self._action_sat_frac.append(float(sat.mean()))
            if "joint_torque" in transition:
                torque = np.abs(transition["joint_torque"][alive])
                self._torque_max_per_step.append(float(torque.max()))
                self._torque_over_limit_frac.append(float((torque > self.torque_limit_nm).mean()))
            if "roll_pitch" in transition:
                rp = np.abs(transition["roll_pitch"][alive])
                self._roll_pitch_max_per_step.append(float(rp.max()))
            if "v_actual" in transition and "v_command" in transition:
                self._v_actual_x.append(float(transition["v_actual"][alive, 0].mean()))
                self._v_command_x.append(float(transition["v_command"][alive, 0].mean()))
                self._tracking_err.append(
                    float(np.abs(transition["v_actual"][alive, 0] - transition["v_command"][alive, 0]).mean())
                )
        if "undesired_contact_count" in transition:
            self._undesired_contact.append(np.asarray(transition["undesired_contact_count"]))

        newly_done = done & (self.fall_step < 0)
        self.fall_step[newly_done] = self._step
        self._step += 1

    def summary(self) -> dict:
        steps = self._step
        never_fell = self.fall_step < 0
        fall_step_capped = np.where(never_fell, steps, self.fall_step)
        out = {
            "steps": steps,
            "mean_survival": float(fall_step_capped.mean()),
            "median_survival": float(np.median(fall_step_capped)),
            "pct_full_survival": float(never_fell.mean() * 100.0),
            "mean_action_saturation_pct": float(np.mean(self._action_sat_frac) * 100.0) if self._action_sat_frac else None,
        }
        if self._torque_max_per_step:
            out["max_torque_nm"] = float(np.max(self._torque_max_per_step))
            out["mean_torque_over_limit_pct"] = float(np.mean(self._torque_over_limit_frac) * 100.0)
        if self._roll_pitch_max_per_step:
            out["max_roll_or_pitch_rad"] = float(np.max(self._roll_pitch_max_per_step))
        if self._v_actual_x:
            mean_v_actual = float(np.mean(self._v_actual_x))
            mean_v_command = float(np.mean(self._v_command_x))
            out["mean_v_actual_x"] = mean_v_actual
            out["mean_v_command_x"] = mean_v_command
            out["tracking_ratio_pct"] = (mean_v_actual / mean_v_command * 100.0) if mean_v_command else 0.0
            out["mean_tracking_error"] = float(np.mean(self._tracking_err))
        if self._undesired_contact:
            contacts = np.array(self._undesired_contact)
            out["mean_undesired_contact_count"] = float(contacts.mean())
            out["frac_steps_with_undesired_contact"] = float(np.mean(contacts > 0))
        return out

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n=== survival ===")
        print(f"mean_survival={s['mean_survival']:.2f}/{s['steps']} median={s['median_survival']:.1f} "
              f"pct_full={s['pct_full_survival']:.1f}%")

        if s.get("mean_action_saturation_pct") is not None:
            print(f"\n=== action saturation (fraction of joints within 5% of ACTION_CLIP={self.action_clip}) ===")
            print(f"mean over rollout: {s['mean_action_saturation_pct']:.1f}%")

        if "max_torque_nm" in s:
            print(f"\n=== joint torque vs A1 limit ({self.torque_limit_nm} Nm) ===")
            print(f"max torque seen: {s['max_torque_nm']:.1f} Nm  "
                  f"mean fraction over limit: {s['mean_torque_over_limit_pct']:.2f}%")

        if "max_roll_or_pitch_rad" in s:
            print(f"\n=== roll/pitch ===")
            print(f"max |roll or pitch|: {s['max_roll_or_pitch_rad']:.3f} rad "
                  f"({np.degrees(s['max_roll_or_pitch_rad']):.1f} deg)")

        if "tracking_ratio_pct" in s:
            print(f"\n=== velocity tracking ===")
            print(f"mean v_actual_x: {s['mean_v_actual_x']:.4f}  mean v_command_x: {s['mean_v_command_x']:.4f}  "
                  f"tracking_ratio: {s['tracking_ratio_pct']:.1f}%")
            print(f"mean |v_actual-v_command|: {s['mean_tracking_error']:.4f}")

        if "mean_undesired_contact_count" in s:
            print(f"\n=== undesired (non-foot) contact ===")
            print(f"mean count: {s['mean_undesired_contact_count']:.4f}  "
                  f"fraction of steps with any: {s['frac_steps_with_undesired_contact']:.4f}")

    def save_plots(self, out_dir: str) -> None:
        """One PNG per per-step series already collected in record() — same
        pattern as Analyzer.save_plots, alongside it under --plot, so a
        --validate run gets a visual trend (does saturation/tracking
        improve or worsen mid-rollout) instead of just one end-of-rollout
        number that hides everything in between."""
        import matplotlib
        matplotlib.use("Agg")  # headless — no display needed to write PNGs
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)

        # v_actual_x/v_command_x plotted together (one figure, two lines)
        # rather than as separate PNGs -- the whole point is comparing
        # tracking against the target, which is hard to eyeball across two
        # separately-scaled plots.
        if self._v_actual_x and self._v_command_x:
            fig, ax = plt.subplots()
            ax.plot(self._v_actual_x, label="v_actual_x")
            ax.plot(self._v_command_x, label="v_command_x", linestyle="--")
            ax.set_title("velocity tracking")
            ax.set_xlabel("step (alive lanes only)")
            ax.legend()
            fig.savefig(os.path.join(out_dir, "validate_velocity_tracking.png"))
            plt.close(fig)

        series = {
            "action_saturation": self._action_sat_frac,
            "torque_max": self._torque_max_per_step,
            "torque_over_limit_frac": self._torque_over_limit_frac,
            "roll_pitch_max": self._roll_pitch_max_per_step,
            "tracking_error": self._tracking_err,
        }
        if self._undesired_contact:
            # (T, N) -- mean across lanes per step, same reduction as the
            # summary's own mean_undesired_contact_count.
            series["undesired_contact_count"] = list(np.array(self._undesired_contact).mean(axis=-1))

        for name, values in series.items():
            if not values:
                continue
            fig, ax = plt.subplots()
            ax.plot(values)
            ax.set_title(name)
            ax.set_xlabel("step (alive lanes only)" if name not in ("undesired_contact_count",) else "step")
            fig.savefig(os.path.join(out_dir, f"validate_{name}.png"))
            plt.close(fig)
