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
