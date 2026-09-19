"""Running per-objective reward normalization (chapter3.tex §3.2.3 —
"prevent a large-magnitude term from dominating other terms' gradient").
Welford's batched-update algorithm (Chan et al. 1979) so stats stay exact
across many small update() calls instead of only a single one-shot batch —
see scripts/rl/core/algorithms/moppo.py for how MOPPO calls this per step.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Tracks a running mean/variance per dimension. `normalize(center=...)`
    defaults to NOT mean-centering (divide by std only), originally so a
    bounded term's 0-boundary would keep its meaning — see
    docs/superpowers/specs for that reasoning. Found 2026-09-18: neither of
    this class's two real call sites (MOPPOTrainer's reward normalization
    and its extrinsics normalization) actually want that default anymore;
    both now pass center=True explicitly (see normalize()'s docstring for
    each incident). The uncentered default survives only as a safe
    fallback for a hypothetical future bounded-[0,1]-term use case that
    doesn't exist in this codebase yet -- nothing currently relies on it."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)  # unit variance until the first update()
        self.count = 0.0  # starts at exactly 0 so the first update() sets exact batch stats

    def update(self, x: np.ndarray) -> None:
        """x: (batch, dim)."""
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0, center: bool = False) -> np.ndarray:
        """center=False (default): divide by std only, preserving a bounded
        term's 0-boundary. Not actually used by either of this class's real
        call sites (see class docstring) -- kept as the default only for a
        hypothetical future bounded-term use case.

        center=True, MOPPOTrainer's REWARD normalization (chapter3.tex Sec
        3.2.3's original use of this class) -- switched 2026-09-18, on the
        same day as the extrinsics fix below, after tracing through what an
        uncentered reward actually does downstream: it's fed straight into
        GAE (gae_per_objective), which ACCUMULATES a constant additive bias
        (mean/std, baked into every uncentered sample) over roughly
        1/(1-gamma*lambda) steps of lookahead (~19 steps at this task's
        gamma=0.998, lambda=0.95) -- but that accumulation isn't uniform
        across a rollout: it's cut short at episode boundaries and at the
        edge of the num_steps rollout window, so samples near either
        boundary carry LESS accumulated bias than samples mid-episode.
        MOPPOTrainer.update()'s later normalize_per_objective only
        subtracts the BATCH-AVERAGE of that position-dependent bias, not
        its sample-to-sample variation -- leaving a spurious
        "how-close-to-a-boundary" signal baked into the advantage that has
        nothing to do with policy quality. Measured on this reward vector:
        energy's raw running mean/std ratio alone was -1.06, comparable in
        size to the term's own std, not a rounding-error-sized effect.

        center=True, MOPPOTrainer's extrinsics e_t (2026-09-18, found
        first, same day) -- several extrinsics channels have
        a large nonzero mean with no special zero-meaning (e.g. motor
        stiffness Kp, running mean ~55 to match RMA's own Kp=55), so
        dividing by std alone left them permanently near +/-`clip` every
        single step regardless of training progress. That constant
        near-clip bias fed into EnvFactorEncoder (env_factor_encoder.py,
        whose final layer has no bounding activation) and came out the
        other side as an oversized z_t (observed up to ~12 in a physics
        validation rollout, runs/phase1_longrun5_2026-09-18/validation/),
        saturating ActorCritic's tanh output on ~86% of joints almost every
        step -- the actual reason every env fell at nearly the same step
        (11-13/60), not organic loss-of-balance. A zero-variance channel
        (this run's leg_length_scale, always exactly 1.0 -- domain
        randomization for it isn't actually varying) is also naturally
        fixed by centering: (x-mean)/std collapses to exactly 0 instead of
        std~=1e-4 blowing x/std up to the clip boundary."""
        std = np.sqrt(self.var + 1e-8)
        numerator = (x - self.mean) if center else x
        return np.clip(numerator / std, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        # Plain Python lists/floats, not numpy arrays — torch.load's default
        # weights_only=True (PyTorch 2.6+) rejects numpy's unpickling globals.
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = state["count"]
