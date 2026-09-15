"""Running per-objective reward normalization (chapter3.tex §3.2.3 —
"prevent a large-magnitude term from dominating other terms' gradient").
Welford's batched-update algorithm (Chan et al. 1979) so stats stay exact
across many small update() calls instead of only a single one-shot batch —
see scripts/rl/core/algorithms/moppo.py for how MOPPO calls this per step.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Tracks a running mean/variance per dimension. `normalize()` scales by
    the running std only (no mean-centering) so a bounded term's 0-boundary
    keeps its meaning — see docs/superpowers/specs for the reasoning."""

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

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        return np.clip(x / std, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        # Plain Python lists/floats, not numpy arrays — torch.load's default
        # weights_only=True (PyTorch 2.6+) rejects numpy's unpickling globals.
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = state["count"]
