# Vendor ConstraintManager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor a cleaned-up `ConstraintManager` + `ConstraintTermCfg` from jaykorea/Isaac-RL-Two-wheel-Legged-Bot into `talon_rl.isaaclab.managers`, importable and unit-testable without Isaac Sim installed, with no env wired to use it yet.

**Architecture:** Two new modules under a new `talon_rl/isaaclab/managers/` package — `constraint_term_cfg.py` (a small `@configclass` dataclass) and `constraint_manager.py` (an Isaac Lab `ManagerBase` subclass implementing the constraint-violation → stochastic-termination-probability mechanism). Both depend on Isaac Lab's real `isaaclab.managers`/`isaaclab.utils` at import time, so tests fake those modules via `sys.modules` injection to exercise the logic without Isaac Sim installed.

**Tech Stack:** Python 3.12, PyTorch, pytest, prettytable (new dependency).

**Spec:** [docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md](../specs/2026-09-15-constraint-manager-vendor-design.md)

## Global Constraints

- Vendor-only scope: do NOT create `ManagerBasedConstraintRLEnv` or wire this manager into any env — no consumer exists yet, per the spec.
- Package layout is single-level `talon_rl/isaaclab/managers/`, not the source project's nested `isaaclab/isaaclab/` (that nesting served a namespace-shadowing need this repo doesn't have).
- All comments in English, explaining why not what.
- Do not carry over the vendored source's `dones` / `get_termination_probs` properties (marked `#* not using yet` upstream — dead code).
- Fix the vendored source's `_class_constraint_cfgs`/`_class_term_cfgs` name-mismatch bug (would `AttributeError` the first time a constraint term's `func` is a `ManagerTermBase` instance).
- Hyperparameters (`tau`, `min_p`, `num_transitions_per_env`, `max_iterations`, `static_curriculum_steps`) are `ConstraintManager.__init__` keyword arguments with the vendored source's values as defaults, not hardcoded body assignments.
- `prettytable` must be added to `pyproject.toml`'s `[project] dependencies`.
- Tests must pass in this repo's default 3.12 `.venv`, which has no Isaac Sim/Isaac Lab installed — achieved by faking `isaaclab.managers.manager_base`, `isaaclab.managers.manager_term_cfg`, and `isaaclab.utils` via `sys.modules` injection, not by skipping.

---

### Task 1: `talon_rl.isaaclab.managers` package + `ConstraintTermCfg`

**Files:**
- Create: `talon_rl/isaaclab/__init__.py`
- Create: `talon_rl/isaaclab/managers/__init__.py`
- Create: `talon_rl/isaaclab/managers/constraint_term_cfg.py`
- Create: `tests/isaaclab/test_constraint_manager.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `talon_rl.isaaclab.managers.constraint_term_cfg.ConstraintTermCfg` — dataclass with fields `func: Callable`, `params: dict` (inherited), `p_max: float = 1.0`, `use_curriculum: bool = False`, `time_out: str = "terminate"`.
- Produces (test infra, reused by Task 2): `tests/isaaclab/test_constraint_manager.py`'s `_install_fake_isaaclab(monkeypatch) -> type` (returns the fake `ManagerTermBase` class), the `constraint_manager_module` fixture (returns `(constraint_manager_module, constraint_term_cfg_module, ManagerTermBase)`), and the `FakeEnv` class (`__init__(self, num_envs, common_step_counter=0)`, attributes `num_envs`, `device`, `common_step_counter`).

- [ ] **Step 1: Add `prettytable` to `pyproject.toml` dependencies**

Edit the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
    "gymnasium>=1.1",
    "prettytable>=3.0",
]
```

- [ ] **Step 2: Write the failing test file with the fake-isaaclab harness and one `ConstraintTermCfg` test**

Create `tests/isaaclab/test_constraint_manager.py`:

```python
# tests/isaaclab/test_constraint_manager.py
"""Unit tests for talon_rl.isaaclab.managers — fakes isaaclab's manager
base classes via sys.modules injection so these run without Isaac Sim
installed (unlike test_a1_env.py, which needs the real thing and skips).
See docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
import torch


def _install_fake_isaaclab(monkeypatch):
    """Injects minimal fake isaaclab.managers/isaaclab.utils modules into
    sys.modules, then drops any cached talon_rl.isaaclab.managers modules
    so the next import binds against the fakes. Returns the fake
    ManagerTermBase class so tests can subclass it."""
    isaaclab_mod = types.ModuleType("isaaclab")
    managers_mod = types.ModuleType("isaaclab.managers")
    utils_mod = types.ModuleType("isaaclab.utils")
    manager_base_mod = types.ModuleType("isaaclab.managers.manager_base")
    manager_term_cfg_mod = types.ModuleType("isaaclab.managers.manager_term_cfg")

    class ManagerTermBase:
        """Fake stand-in for isaaclab.managers.manager_base.ManagerTermBase
        — the real base class for stateful constraint-term callables."""

        def __init__(self, cfg=None, env=None):
            self.cfg = cfg
            self._env = env

        def reset(self, env_ids=None):
            pass

    class ManagerBase:
        """Fake stand-in for isaaclab.managers.manager_base.ManagerBase —
        just the __init__/_prepare_terms wiring ConstraintManager needs."""

        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device
            self._prepare_terms()

        def _resolve_common_term_cfg(self, term_name, term_cfg, min_argc=1):
            pass

    manager_base_mod.ManagerBase = ManagerBase
    manager_base_mod.ManagerTermBase = ManagerTermBase

    @dataclass
    class ManagerTermBaseCfg:
        """Fake stand-in for
        isaaclab.managers.manager_term_cfg.ManagerTermBaseCfg."""

        func: Callable = None
        params: dict[str, Any] = field(default_factory=dict)
        time_out: bool = False

    manager_term_cfg_mod.ManagerTermBaseCfg = ManagerTermBaseCfg

    def configclass(cls):
        """Fake stand-in for isaaclab.utils.configclass — the real one adds
        dataclass ergonomics irrelevant to these tests."""
        return dataclass(cls)

    utils_mod.configclass = configclass

    fake_modules = {
        "isaaclab": isaaclab_mod,
        "isaaclab.managers": managers_mod,
        "isaaclab.managers.manager_base": manager_base_mod,
        "isaaclab.managers.manager_term_cfg": manager_term_cfg_mod,
        "isaaclab.utils": utils_mod,
    }
    for name, mod in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    for name in [
        "talon_rl.isaaclab",
        "talon_rl.isaaclab.managers",
        "talon_rl.isaaclab.managers.constraint_manager",
        "talon_rl.isaaclab.managers.constraint_term_cfg",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    return ManagerTermBase


@pytest.fixture
def constraint_manager_module(monkeypatch):
    """Imports talon_rl.isaaclab.managers' constraint modules with a fake
    isaaclab installed, so they're unit-testable without Isaac Sim."""
    manager_term_base_cls = _install_fake_isaaclab(monkeypatch)
    term_cfg_module = importlib.import_module(
        "talon_rl.isaaclab.managers.constraint_term_cfg"
    )
    manager_module = importlib.import_module(
        "talon_rl.isaaclab.managers.constraint_manager"
    )
    return manager_module, term_cfg_module, manager_term_base_cls


class FakeEnv:
    """Minimal stand-in for a ManagerBasedRLEnv — just the attributes
    ConstraintManager reads (num_envs, device, common_step_counter)."""

    def __init__(self, num_envs: int, common_step_counter: int = 0):
        self.num_envs = num_envs
        self.device = "cpu"
        self.common_step_counter = common_step_counter


def test_constraint_term_cfg_constructs_with_required_func(constraint_manager_module):
    _, term_cfg_module, _ = constraint_manager_module

    def violation(env):
        return torch.zeros(env.num_envs)

    cfg = term_cfg_module.ConstraintTermCfg(func=violation, time_out="constraint", p_max=0.5)

    assert cfg.func is violation
    assert cfg.p_max == 0.5
    assert cfg.use_curriculum is False
    assert cfg.time_out == "constraint"
```

Note: `talon_rl/isaaclab/managers/constraint_manager.py` does not exist yet
in this task — the test file already references it (inside the fixture,
executed lazily per-test) so Task 2 can add tests to this same file without
touching the fixture. This task's own test only exercises
`constraint_term_cfg`, so it will fail at collection/run because
`talon_rl.isaaclab` doesn't exist at all yet — that's the expected failure
this task's implementation step fixes.

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/isaaclab/test_constraint_manager.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'talon_rl.isaaclab'`

- [ ] **Step 4: Create the package skeleton**

Create `talon_rl/isaaclab/__init__.py`:

```python
"""Namespace for talon_rl code that extends Isaac Lab's own core
framework (managers, envs) rather than building on top of it — as opposed
to talon_rl.tasks / talon_rl.assets, which use stock Isaac Lab classes
unmodified. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md.
"""
```

Create `talon_rl/isaaclab/managers/__init__.py`:

```python
"""Constraint-based termination manager, vendored from
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
lab/flamingo/isaaclab/isaaclab/managers/. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md for
what changed from the vendored source and why. Not wired into any env yet
— ConstraintManager has no consumer in this repo.
"""

from .constraint_term_cfg import ConstraintTermCfg

__all__ = ["ConstraintTermCfg"]
```

- [ ] **Step 5: Write `constraint_term_cfg.py`**

Create `talon_rl/isaaclab/managers/constraint_term_cfg.py`:

```python
# talon_rl/isaaclab/managers/constraint_term_cfg.py
"""Configuration for a constraint term, vendored from
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
lab/flamingo/isaaclab/isaaclab/managers/constraint_term_cfg.py. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

import torch
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.utils import configclass


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    """Configuration for a constraint term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The function to call for this term.

    Must take the environment object and any params in `self.params` as
    input and return a float tensor of shape (num_envs,) giving each env's
    degree of violation in [0, 1].
    """

    p_max: float = 1.0
    """Maximum scaling factor for this constraint's termination probability.

    Use 1.0 for a hard constraint (strictly enforced once the curriculum
    reaches full strength, if any). Use a value below 1.0 for a soft
    constraint that still allows some exploration even at full curriculum
    strength.
    """

    use_curriculum: bool = False
    """Whether to ramp this constraint's enforcement in over training via
    ConstraintManager's curriculum, rather than enforcing at full strength
    from the first step."""

    time_out: str = "terminate"
    """How this term's violation value is consumed: "truncate" or
    "terminate" for a hard binary 0/1 termination signal, or "constraint"
    for ConstraintManager's soft, curriculum-scaled stochastic termination
    probability. Reuses ManagerTermBaseCfg's `time_out` field name (there a
    plain bool) with three-way string semantics instead — kept for
    Isaac Lab API-shape compatibility rather than renamed to a
    constraint-specific field.
    """
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/isaaclab/test_constraint_manager.py -v`
Expected: PASS — `test_constraint_term_cfg_constructs_with_required_func` passes.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml talon_rl/isaaclab tests/isaaclab/test_constraint_manager.py
git commit -m "$(cat <<'EOF'
feat: vendor ConstraintTermCfg into talon_rl.isaaclab.managers

First half of vendoring jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
ConstraintManager mechanism for future use — no env consumes it yet.
Adds the fake-isaaclab test harness reused by ConstraintManager's tests.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `ConstraintManager`

**Files:**
- Create: `talon_rl/isaaclab/managers/constraint_manager.py`
- Modify: `talon_rl/isaaclab/managers/__init__.py`
- Modify: `tests/isaaclab/test_constraint_manager.py`

**Interfaces:**
- Consumes: `talon_rl.isaaclab.managers.constraint_term_cfg.ConstraintTermCfg` (Task 1). Test fixtures `constraint_manager_module`, `FakeEnv`, and helper `_install_fake_isaaclab` (Task 1, already in `tests/isaaclab/test_constraint_manager.py` — reused unchanged).
- Produces: `talon_rl.isaaclab.managers.constraint_manager.ConstraintManager` — constructor `ConstraintManager(cfg, env, *, tau=0.95, min_p=0.0, num_transitions_per_env=24, max_iterations=5000, static_curriculum_steps=30000)`; methods `compute() -> torch.Tensor`, `reset(env_ids=None) -> dict[str, torch.Tensor]`, `get_term(name) -> torch.Tensor`, `set_term_cfg(term_name, cfg)`, `get_term_cfg(term_name) -> ConstraintTermCfg`, `get_active_iterable_terms(env_idx) -> Sequence[tuple[str, Sequence[float]]]`; properties `active_terms`, `time_outs`, `constrained`, `hard_constrained`.

- [ ] **Step 1: Append the failing `ConstraintManager` tests**

Append to `tests/isaaclab/test_constraint_manager.py`:

```python
def test_constraint_probability_stays_within_p_max(constraint_manager_module):
    manager_module, term_cfg_module, _ = constraint_manager_module
    env = FakeEnv(num_envs=3)

    def torque_violation(env):
        return torch.tensor([0.0, 0.5, 1.0])

    cfg = types.SimpleNamespace(
        torque=term_cfg_module.ConstraintTermCfg(
            func=torque_violation, time_out="constraint", p_max=0.5, use_curriculum=False
        )
    )
    manager = manager_module.ConstraintManager(cfg, env)
    manager.compute()

    assert torch.all(manager.constrained >= 0.0)
    assert torch.all(manager.constrained <= 0.5 + 1e-6)
    torch.testing.assert_close(manager.constrained, torch.tensor([0.0, 0.25, 0.5]))


def test_curriculum_forces_zero_before_threshold_and_ramps_after(constraint_manager_module):
    manager_module, term_cfg_module, _ = constraint_manager_module

    def torque_violation(env):
        return torch.tensor([1.0])

    cfg = types.SimpleNamespace(
        torque=term_cfg_module.ConstraintTermCfg(
            func=torque_violation, time_out="constraint", p_max=0.5, use_curriculum=True
        )
    )

    env_before = FakeEnv(num_envs=1, common_step_counter=0)
    manager_before = manager_module.ConstraintManager(cfg, env_before, static_curriculum_steps=100)
    manager_before.compute()
    assert torch.all(manager_before.constrained == 0.0)

    env_after = FakeEnv(num_envs=1, common_step_counter=200)
    manager_after = manager_module.ConstraintManager(cfg, env_after, static_curriculum_steps=100)
    manager_after.compute()
    assert torch.all(manager_after.constrained > 0.0)


def test_manager_term_base_func_does_not_raise_on_construction(constraint_manager_module):
    manager_module, term_cfg_module, ManagerTermBase = constraint_manager_module

    class StatefulTerm(ManagerTermBase):
        def __call__(self, env):
            return torch.zeros(env.num_envs)

    term_instance = StatefulTerm()
    env = FakeEnv(num_envs=2)
    cfg = types.SimpleNamespace(
        stateful=term_cfg_module.ConstraintTermCfg(func=term_instance, time_out="terminate")
    )

    manager = manager_module.ConstraintManager(cfg, env)  # must not raise

    assert len(manager._class_term_cfgs) == 1
    manager.reset()  # must not raise — calls term_instance.reset(env_ids=...)


def test_truncate_term_rejects_non_binary_values(constraint_manager_module):
    manager_module, term_cfg_module, _ = constraint_manager_module

    def bad_violation(env):
        return torch.tensor([0.3, 1.0])

    env = FakeEnv(num_envs=2)
    cfg = types.SimpleNamespace(
        limit=term_cfg_module.ConstraintTermCfg(func=bad_violation, time_out="truncate")
    )
    manager = manager_module.ConstraintManager(cfg, env)

    with pytest.raises(ValueError):
        manager.compute()


def test_terminate_term_rejects_non_binary_values(constraint_manager_module):
    manager_module, term_cfg_module, _ = constraint_manager_module

    def bad_violation(env):
        return torch.tensor([0.0, 0.5])

    env = FakeEnv(num_envs=2)
    cfg = types.SimpleNamespace(
        fall=term_cfg_module.ConstraintTermCfg(func=bad_violation, time_out="terminate")
    )
    manager = manager_module.ConstraintManager(cfg, env)

    with pytest.raises(ValueError):
        manager.compute()
```

(`types` is already imported at the top of `tests/isaaclab/test_constraint_manager.py`
from Task 1 — no new imports needed for this step.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/isaaclab/test_constraint_manager.py -v`
Expected: the 5 new tests FAIL/ERROR — `ModuleNotFoundError: No module named 'talon_rl.isaaclab.managers.constraint_manager'`. The Task 1 test still passes.

- [ ] **Step 3: Write `constraint_manager.py`**

Create `talon_rl/isaaclab/managers/constraint_manager.py`:

```python
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

        self.env = env

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
        """Computes the stochastic termination signal based on constraint violations.

        Returns:
            A bool tensor of shape (num_envs,) — True where an env should
            terminate this step.
        """
        self._truncated_buf.zero_()
        self._delta_buf.zero_()

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params).float()

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)

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
                    if self.env.common_step_counter < self.static_curriculum_steps:
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
```

- [ ] **Step 4: Update the package `__init__.py`**

Edit `talon_rl/isaaclab/managers/__init__.py`:

```python
"""Constraint-based termination manager, vendored from
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
lab/flamingo/isaaclab/isaaclab/managers/. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md for
what changed from the vendored source and why. Not wired into any env yet
— ConstraintManager has no consumer in this repo.
"""

from .constraint_manager import ConstraintManager
from .constraint_term_cfg import ConstraintTermCfg

__all__ = ["ConstraintManager", "ConstraintTermCfg"]
```

- [ ] **Step 5: Run the full test file to verify everything passes**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/isaaclab/test_constraint_manager.py -v`
Expected: PASS — all 6 tests (1 from Task 1, 5 from this task).

- [ ] **Step 6: Run the full unaffected suite to confirm no regressions**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/rewards/test_locomotion.py tests/core/preferences/test_preferences.py tests/isaaclab/test_constraint_manager.py -v`
Expected: PASS — 11 pre-existing + 6 new = 17 passed. (`test_dummy_env.py`/`test_moppo_smoke.py` continue to need `gymnasium` per this environment's pre-existing gap, unrelated to this change; `test_a1_env.py` continues to skip without Isaac Sim.)

- [ ] **Step 7: Commit**

```bash
git add talon_rl/isaaclab/managers/constraint_manager.py talon_rl/isaaclab/managers/__init__.py tests/isaaclab/test_constraint_manager.py
git commit -m "$(cat <<'EOF'
feat: vendor ConstraintManager into talon_rl.isaaclab.managers

Completes vendoring jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
ConstraintManager for future use. Fixes a latent bug in the vendored
source (_class_constraint_cfgs/_class_term_cfgs name mismatch that
would AttributeError the first time a constraint term's func is a
ManagerTermBase instance) and moves hardcoded hyperparameters to
constructor kwargs. Still vendor-only — no env consumes this yet.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
