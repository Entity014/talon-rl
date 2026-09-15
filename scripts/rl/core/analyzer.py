"""Per-step signal collection + plotting for play.py's --analyze/--plot —
own implementation built on the transition dict's own fields (v_actual,
v_command, joint_torque, joint_vel, foot_contact_force — see
talon_rl/envs/base_env.py's documented transition contract), NOT a port of
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's Analyzer: that one introspects
Isaac Lab's observation_manager internals directly (active_terms,
scene["robot"].joint_names) which don't exist on DummyTalonEnv, and this
repo's IsaacLabTalonEnv bypasses Isaac Lab's reward/observation manager
entirely (see a1_env_cfg.py's own docstring) — there is no
observation_manager to introspect here even on the real env.
"""

from __future__ import annotations

import os

import numpy as np


class Analyzer:
    """Collects one array per requested transition-dict key across a
    play.py rollout, then saves one PNG time-series plot per key."""

    def __init__(self, items: list[str]):
        self.items = items
        self._data: dict[str, list[np.ndarray]] = {item: [] for item in items}

    def record(self, transition: dict) -> None:
        for item in self.items:
            if item not in transition:
                raise KeyError(f"'{item}' not in this env's transition dict — available keys: {list(transition)}")
            self._data[item].append(np.asarray(transition[item]))

    def save_plots(self, out_dir: str) -> None:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display needed to write PNGs
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        for item, frames in self._data.items():
            series = np.stack(frames)  # (T, N) or (T, N, D)
            # Plot lane 0 only — one representative env, not all N (a
            # multi-thousand-env plot would be unreadable regardless).
            lane0 = series[:, 0]
            fig, ax = plt.subplots()
            if lane0.ndim == 1:
                ax.plot(lane0)
            else:
                for d in range(lane0.shape[-1]):
                    ax.plot(lane0[:, d], label=f"{item}[{d}]")
                ax.legend()
            ax.set_title(item)
            ax.set_xlabel("step")
            fig.savefig(os.path.join(out_dir, f"{item}.png"))
            plt.close(fig)
