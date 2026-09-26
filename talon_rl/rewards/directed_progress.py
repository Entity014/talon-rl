"""Pure, simulator-agnostic logic for progress_reward's directed-progress
term (R1, 2026-09-20 -- see artifacts/r1_freeze/FREEZE.md for the frozen
equation/constants this implements).

Deliberately split from talon_rl/tasks/locomotion/a1_env/a1_env.py (the only
caller today): that module is Isaac Lab/GPU-gated and its own test
(tests/test_a1_env.py) silently kills the whole pytest process on this
machine, so any logic living only there is effectively untestable here.
Everything that can be expressed as plain numpy on (N, ...) arrays --
window bookkeeping, reset-on-command-change, heading projection, clipping --
lives here instead, where tests/test_directed_progress.py can exercise it
directly. a1_env.py's job is reduced to: pull root position / yaw / command
/ done out of Isaac Lab, hand them to update(), store the returned state,
report the returned directed_progress. No Isaac Lab types appear below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 0.5s at dt=0.01 (frozen -- see FREEZE.md). Not swept, not derived from a
# live step counter: the formula divides by this constant directly, it does
# not require literally counting 50 elapsed steps (a lane's displacement
# keeps accumulating past T_w and the ratio saturates at 1, which is the
# intended "sustained credit once you've covered the expected distance"
# behavior, not a sliding window).
WINDOW_SECONDS = 0.5


@dataclass
class DirectedProgressState:
    """Per-lane window-start snapshot. `origin_xy`/`origin_heading` are the
    position/heading the current window's displacement is measured from;
    `origin_command_x` is the last commanded v_x seen, used only to detect a
    command change (not part of the reward formula itself). All (N, ...),
    numpy, float32/float64 -- caller's dtype choice, passed through as-is."""

    origin_xy: np.ndarray  # (N, 2)
    origin_heading: np.ndarray  # (N,)
    origin_command_x: np.ndarray  # (N,)

    @classmethod
    def initial(cls, root_xy: np.ndarray, heading: np.ndarray, command_x: np.ndarray) -> "DirectedProgressState":
        """A window starting THIS step, as if every lane had just reset --
        used at construction and wherever a caller doesn't have a prior
        state yet."""
        return cls(origin_xy=np.array(root_xy, copy=True), origin_heading=np.array(heading, copy=True),
                    origin_command_x=np.array(command_x, copy=True))


def update(
    state: DirectedProgressState,
    root_xy: np.ndarray, heading: np.ndarray, command_x: np.ndarray,
    reset_mask: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, DirectedProgressState]:
    """One step. Returns (directed_progress (N,) in [0, 1], new_state).

    Window reset triggers (per FREEZE.md -- must NOT carry state across any
    of these, and must NOT use future/privileged info):
    - `reset_mask[i]` True: caller says lane i's episode reset this step
      (includes "lane terminated", same signal -- a terminated lane gets a
      fresh episode next step through this same mask, not a separate path).
    - `command_x[i] != state.origin_command_x[i]` (beyond `eps`): the
      command changed underneath the lane -- last window's progress no
      longer describes progress toward the CURRENT command, so it doesn't
      carry over.

    On a reset step, the window starts THIS instant (origin := current
    root_xy/heading/command_x) and directed_progress reads exactly 0 for
    that lane (zero elapsed displacement) -- this is what keeps a reset or
    command-change step from ever producing a spike; there is no separate
    "don't compute this step" branch, the formula's own zero-displacement
    case already produces zero.

    Projection uses `state.origin_heading` (heading AT WINDOW START), never
    the current heading -- a lane that's since turned must still be scored
    against the direction it was facing when the command took effect, not
    whatever direction it happens to face now.
    """
    root_xy = np.asarray(root_xy, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    command_x = np.asarray(command_x, dtype=np.float64)
    reset_mask = np.asarray(reset_mask, dtype=bool)

    command_changed = np.abs(command_x - state.origin_command_x) > eps
    do_reset = reset_mask | command_changed

    origin_xy = np.where(do_reset[:, None], root_xy, state.origin_xy)
    origin_heading = np.where(do_reset, heading, state.origin_heading)
    origin_command_x = np.where(do_reset, command_x, state.origin_command_x)

    delta = root_xy - origin_xy
    dx_heading = delta[:, 0] * np.cos(origin_heading) + delta[:, 1] * np.sin(origin_heading)

    zero_cmd = np.abs(command_x) < eps
    denom = np.abs(command_x) * WINDOW_SECONDS + eps
    ratio = np.sign(command_x) * dx_heading / denom
    directed_progress = np.where(zero_cmd, 0.0, np.clip(ratio, 0.0, 1.0)).astype(np.float32)

    new_state = DirectedProgressState(origin_xy=origin_xy, origin_heading=origin_heading, origin_command_x=origin_command_x)
    return directed_progress, new_state
