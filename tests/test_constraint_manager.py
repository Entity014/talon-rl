# tests/test_constraint_manager.py
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


def test_constraint_term_converts_plain_python_values_to_tensor(constraint_manager_module):
    manager_module, term_cfg_module, _ = constraint_manager_module

    # Test with plain Python list
    def list_violation(env):
        return [0.0, 0.5, 1.0]

    env = FakeEnv(num_envs=3)
    cfg = types.SimpleNamespace(
        list_term=term_cfg_module.ConstraintTermCfg(
            func=list_violation, time_out="constraint", p_max=0.5, use_curriculum=False
        )
    )
    manager = manager_module.ConstraintManager(cfg, env)
    manager.compute()

    # Should have converted the list to tensor and computed probabilities
    assert torch.is_tensor(manager.constrained)
    assert manager.constrained.shape == torch.Size([3])
    torch.testing.assert_close(manager.constrained, torch.tensor([0.0, 0.25, 0.5]))
