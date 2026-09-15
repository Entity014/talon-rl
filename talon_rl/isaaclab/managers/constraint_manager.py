# talon_rl/isaaclab/managers/constraint_manager.py
"""Manager for computing constraint violation signals, vendored from
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
lab/flamingo/isaaclab/isaaclab/managers/constraint_manager.py. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md for
what changed from the vendored source and why. Not wired into any env yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase

from .constraint_term_cfg import ConstraintTermCfg

if TYPE_CHECKING:
    # Points at the stock Isaac Lab env class since no TALON env subclass
    # consumes this manager yet — repoint this at that subclass's type once
    # one exists (see the vendor design doc's out-of-scope section).
    from isaaclab.envs import ManagerBasedRLEnv


class ConstraintManager(ManagerBase):
    """Manager for computing continuous constraint violation signals.

    Each constraint term is a function that takes the environment as an
    argument and returns a float tensor of shape (num_envs,) in [0, 1]
    representing the degree of violation. The overall constraint signal is
    computed as the element-wise maximum over the individual term signals,
    with terms flagged time_out="truncate" stored in one buffer and the
    remaining terms in another.
    """

    _env: ManagerBasedRLEnv

    def __init__(
        self,
        cfg: object,
        env: ManagerBasedRLEnv,
        *,
        tau: float = 0.95,
        min_p: float = 0.0,
        num_transitions_per_env: int = 24,
        max_iterations: int = 5000,
        static_curriculum_steps: int = 30000,
    ):
        """Initializes the constraint manager.

        Args:
            cfg: The configuration object or dictionary for constraint
                terms, where each term should be an instance of
                ConstraintTermCfg.
            env: An environment object.
            tau: Exponential-moving-average coefficient for each
                "constraint"-mode term's running max violation.
            min_p: Minimum termination probability for a violated
                "constraint"-mode term.
            num_transitions_per_env: Rollout length used (with
                max_iterations) to compute the per-step curriculum
                increment.
            max_iterations: Training iterations used (with
                num_transitions_per_env) to compute the per-step curriculum
                increment.
            static_curriculum_steps: Environment steps (per env) before a
                use_curriculum=True term's p_max starts ramping up from 0.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[ConstraintTermCfg] = []
        self._class_term_cfgs: list[ConstraintTermCfg] = []

        super().__init__(cfg, env)  # _prepare_terms() called here

        self.tau = tau
        self.min_p = min_p
        self.num_transitions_per_env = num_transitions_per_env
        self.max_iterations = max_iterations
        self.step_cur = 1.0 / (self.num_transitions_per_env * self.max_iterations)
        self.static_curriculum_steps = static_curriculum_steps

        self._term_values = {}
        self.curriculum = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            self._term_values[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

            if term_cfg.time_out not in ["truncate", "terminate", "constraint"]:
                raise ValueError(f"Invalid time_out value '{term_cfg.time_out}' for term '{term_cfg.func}'.")

            if term_cfg.time_out == "constraint" and term_cfg.use_curriculum:
                self.curriculum[name] = 0.0

        # per-constraint stochastic-termination bookkeeping
        self._running_maxes = {}
        self._probs = {}

        self._truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._delta_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def __str__(self) -> str:
        """Returns a string representation for the constraint manager."""
        msg = f"<ConstraintManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Constraint Terms"
        table.field_names = ["Index", "Name", "p_max"]
        table.align["Name"] = "l"
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, getattr(term_cfg, "p_max", 1.0)])
        msg += table.get_string() + "\n"
        return msg

    @property
    def active_terms(self) -> list[str]:
        """Name of active constraint terms."""
        return self._term_names

    @property
    def time_outs(self) -> torch.Tensor:
        """Returns the timeout signal computed from time_out="truncate" terms."""
        return self._truncated_buf

    @property
    def constrained(self) -> torch.Tensor:
        """Returns the soft constraint signal (delta) from the remaining terms."""
        return self._delta_buf

    @property
    def hard_constrained(self) -> torch.Tensor:
        """Returns which envs hit a constraint signal of exactly 1.0."""
        return self._delta_buf == 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the constraint term values for a new episode and returns summary information.

        Args:
            env_ids: The environment ids to reset (if None, reset all environments).

        Returns:
            A dictionary containing the episodic sums for each constraint term.
        """
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for key, term in zip(self._term_values.keys(), self._term_cfgs):
            if term.time_out in ("truncate", "terminate"):
                extras["Episode_Constraint/" + key] = torch.count_nonzero(
                    self._term_values[key][env_ids].float()
                ).item()
            else:
                extras["Episode_Constraint/" + key] = torch.mean(self._term_values[key][env_ids]).item()

        for key in self._probs.keys():
            self._probs[key][env_ids] = 0.0

        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras

    def compute(self) -> torch.Tensor:
        """Computes the termination mask based on constraint violations.

        This method is deterministic: it returns a bool mask of envs whose
        combined signal saturated at 1.0 (hard violations from truncate/terminate
        terms, or a constraint term whose stochastic probability reached exactly
        1.0). The soft, curriculum-scaled per-env termination probabilities for
        'constraint' terms are exposed via the `constrained` property — sampling
        them into an actual stochastic termination decision is the consuming
        env's responsibility, not this method's.

        Returns:
            A bool tensor of shape (num_envs,) — True where an env's signal
            reached exactly 1.0 and should terminate this step.
        """
        self._truncated_buf.zero_()
        self._delta_buf.zero_()

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)

            value = value.float()

            if term_cfg.time_out == "truncate":
                value = torch.clamp(value, 0.0, 1.0)
                if not torch.all((value == 0.0) | (value == 1.0)):
                    raise ValueError("value must be either 0 or 1.")
                self._truncated_buf = torch.max(self._truncated_buf, value)
                self._term_values[name][:] = value
            elif term_cfg.time_out == "terminate":
                value = torch.clamp(value, 0.0, 1.0)
                if not torch.all((value == 0.0) | (value == 1.0)):
                    raise ValueError("value must be either 0 or 1.")
                self._delta_buf = torch.max(self._delta_buf, value)
                self._term_values[name][:] = value
            elif term_cfg.time_out == "constraint":
                p_max = term_cfg.p_max
                if term_cfg.use_curriculum:
                    if self._env.common_step_counter < self.static_curriculum_steps:
                        p_max = 0.0
                    else:
                        self.curriculum[name] = min(self.curriculum[name] + self.step_cur, 1.0)
                        t_start = 20
                        t_end = max(1.0 / p_max, 1e-6)
                        p_max = 1.0 / (t_start + self.curriculum[name] * (t_end - t_start))

                # this step's own worst violation across the batch, used to
                # normalize the rest of the batch's violations into [0, 1]
                constraint_max = value.max(dim=0, keepdim=True)[0].clamp(min=1e-6)

                if name not in self._running_maxes:
                    self._running_maxes[name] = constraint_max.clone()
                else:
                    self._running_maxes[name] = (
                        self.tau * self._running_maxes[name] + (1.0 - self.tau) * constraint_max
                    )

                mask = value > 0.0
                probs = torch.zeros_like(value, dtype=torch.float32)
                probs[mask] = (
                    self.min_p
                    + torch.clamp(
                        value[mask] / self._running_maxes[name].expand(value.size())[mask],
                        min=0.0,
                        max=1.0,
                    )
                    * (p_max - self.min_p)
                ).to(probs.dtype)

                self._probs[name] = probs
                self._delta_buf = torch.max(self._delta_buf, probs)
                self._term_values[name][:] = value

        reset_buf = torch.max(self._truncated_buf, self._delta_buf)
        return reset_buf == 1.0

    def get_term(self, name: str) -> torch.Tensor:
        """Returns the constraint term value for the specified name."""
        return self._term_values[name]

    def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
        """Sets the configuration for the specified constraint term."""
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        """Gets the configuration for the specified constraint term."""
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns each active constraint term's raw value for one env index."""
        terms = []
        for key in self._term_names:
            terms.append((key, [self._term_values[key][env_idx].float().cpu().item()]))
        return terms

    def _prepare_terms(self):
        """Parses the configuration and prepares the constraint terms."""
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg. "
                    f"Received: '{type(term_cfg)}'."
                )
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
