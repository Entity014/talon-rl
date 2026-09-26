"""Pure G1 command-exposure state machine.

This module has no Isaac/torch dependencies so its boundary semantics can be
tested before wiring it into the environment. A lane's ``step`` is the policy
step about to receive an observation and action.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class G1CommandEvent:
    command: np.ndarray
    zero_hold: np.ndarray
    hold_step: np.ndarray
    progress_reset: np.ndarray


class G1CommandSchedule:
    def __init__(self, lanes: int, rng: np.random.Generator,
                 zero_probability: float = 0.25, hold_steps: int = 200):
        if not 0.0 <= zero_probability <= 1.0:
            raise ValueError("zero_probability must be in [0, 1]")
        self.n = lanes
        self.rng = rng
        self.p = zero_probability
        self.hold_steps = hold_steps
        self.zero_hold = np.zeros(lanes, dtype=bool)
        self.hold_step = np.zeros(lanes, dtype=np.int32)
        self.command = np.zeros((lanes, 3), dtype=np.float32)
        self._reset_all(np.arange(lanes))

    def _sample_command(self, count: int) -> np.ndarray:
        return np.column_stack((
            self.rng.uniform(-0.3, 1.0, count),
            self.rng.uniform(-0.3, 0.3, count),
            self.rng.uniform(-0.5, 0.5, count),
        )).astype(np.float32)

    def _reset_all(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.zero_hold[ids] = self.rng.random(len(ids)) < self.p
        self.hold_step[ids] = 0
        self.command[ids] = 0.0
        active = ~self.zero_hold[ids]
        if active.any():
            self.command[ids[active]] = self._sample_command(int(active.sum()))

    def reset_lanes(self, done: np.ndarray) -> None:
        """Start fresh episodes; terminated lanes never resume an old hold."""
        ids = np.flatnonzero(np.asarray(done, dtype=bool))
        if len(ids):
            self._reset_all(ids)

    def before_observation(self) -> G1CommandEvent:
        """Return command for the next observation; switch occurs at step 200."""
        # Called for the observation that follows the completed physics step:
        # after steps 0..198, counters are 1..199; the call producing obs200
        # switches before returning the observation.
        switch = self.zero_hold & (self.hold_step == self.hold_steps - 1)
        if switch.any():
            ids = np.flatnonzero(switch)
            self.command[ids] = self._sample_command(len(ids))
        event_steps = self.hold_step.copy()
        event_steps[switch] = self.hold_steps
        event = G1CommandEvent(self.command.copy(), self.zero_hold.copy(),
                               event_steps, switch.copy())
        active_hold = self.zero_hold & ~switch
        self.hold_step[active_hold] += 1
        self.hold_step[switch] = self.hold_steps
        return event
