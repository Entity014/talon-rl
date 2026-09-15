# Adaptation Module Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire RMA's Phase 1 (teacher) — a privileged Env Factor Encoder μ that compresses 7 environment/morphology extrinsics into a latent $z_t$ fed into the base policy — into the real Isaac Lab A1 env and MOPPOTrainer.

**Architecture:** Isaac Lab's stock `EventManager` randomizes 6 of the 7 extrinsics (leg-length is spawn-time, via pre-generated USD variants + `MultiUsdFileCfg`, not an event — Isaac Lab blocks runtime scale-randomization on an `Articulation`). A new `Privileged` `ObservationManager` group exposes the raw extrinsics vector $e_t$. `EnvFactorEncoder` (a small `nn.Module` in the trainer layer, never an Isaac Lab Manager — it needs gradient-tracked parameters trained jointly with the policy) encodes $e_t \to z_t$; `MOPPOTrainer` concatenates $z_t$ into the actor's observation the same way the preference vector $w$ already is.

**Tech Stack:** Isaac Lab 0.48.0 (`isaaclab.envs.mdp.events`, `isaaclab.sim.spawners.wrappers.MultiUsdFileCfg`), PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-adaptation-module-phase1-design.md`

## Global Constraints

- All 7 extrinsics factors in scope (mass, CoM, friction, terrain height, motor power, leg length, joint range) — no factor deferred.
- `PolicyCfg` (the deployed policy's observation group) must **never** gain any extrinsics term — Phase 2's whole premise is the deployed policy never sees privileged state.
- The encoder (`EnvFactorEncoder`) lives in `scripts/rl/core/modules/`, never in an Isaac Lab Manager — it must joint-train with the policy through `MOPPOTrainer.optim`.
- `payload_treatment` (`explicit_observed_rewarded` | `noise_only`) changes $e_t$'s width itself (payload excluded entirely under `noise_only`), not just a downstream reward gate.
- `--env dummy` must keep working unmodified — `MOPPOTrainer` must not hard-require an `extrinsics` key in the transition dict.
- Payload reward term's exact formula and randomization parameter ranges beyond payload mass (which the Pipeline_Summary.md Pareto-sweep table already fixes at 0-5 kg) are **explicitly out of scope** per the spec's Open Questions — stub, don't invent.
- TDD required for every step with a `- [ ] Write the failing test` step, per this repo's CLAUDE.md ("every non-obvious decision gets a test, and the test's comment says which failure it exists to prevent").

---

## Task 1: `ExtrinsicsCfg` dataclass

**Files:**
- Modify: `talon_rl/config.py`
- Test: `tests/test_extrinsics_cfg.py`

**Interfaces:**
- Produces: `ExtrinsicsCfg` (frozen dataclass) with fields `payload_mass_dim`, `payload_com_offset_dim`, `friction_dim`, `motor_power_scale_dim`, `leg_length_scale_dim`, `joint_range_scale_dim`, `terrain_height_dim`, `payload_treatment: Literal["explicit_observed_rewarded", "noise_only"]`, `adaptation_latent_dim`; properties `payload_dim` and `dim`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extrinsics_cfg.py
import pytest

from talon_rl.config import ExtrinsicsCfg


def test_payload_dim_sums_mass_and_com_offset():
    cfg = ExtrinsicsCfg()
    assert cfg.payload_dim == cfg.payload_mass_dim + cfg.payload_com_offset_dim


def test_dim_includes_payload_under_explicit_observed_rewarded():
    cfg = ExtrinsicsCfg(payload_treatment="explicit_observed_rewarded")
    non_payload = (
        cfg.friction_dim + cfg.motor_power_scale_dim + cfg.leg_length_scale_dim
        + cfg.joint_range_scale_dim + cfg.terrain_height_dim
    )
    assert cfg.dim == non_payload + cfg.payload_dim


def test_dim_excludes_payload_under_noise_only():
    # noise_only: payload is randomized in sim but never observed by the
    # encoder — e_t's width itself shrinks, not just a reward-term toggle
    # (Pipeline_Summary.md §3.10: Baseline A "ไม่สังเกตหรือให้รางวัลแยก").
    cfg = ExtrinsicsCfg(payload_treatment="noise_only")
    non_payload = (
        cfg.friction_dim + cfg.motor_power_scale_dim + cfg.leg_length_scale_dim
        + cfg.joint_range_scale_dim + cfg.terrain_height_dim
    )
    assert cfg.dim == non_payload


def test_frozen_rejects_mutation():
    cfg = ExtrinsicsCfg()
    with pytest.raises(Exception):
        cfg.payload_mass_dim = 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_extrinsics_cfg.py -v`
Expected: FAIL with `ImportError: cannot import name 'ExtrinsicsCfg'`

- [ ] **Step 3: Write minimal implementation**

Append to `talon_rl/config.py` (add `from typing import Literal` to the existing `from __future__ import annotations` / `from dataclasses import dataclass, field` imports at the top if not already present):

```python
@dataclass(frozen=True)
class ExtrinsicsCfg:
    """Phase 1 privileged extrinsics e_t (chapter3.tex §3.2.1, RMA's Env
    Factor Encoder input). Dimensions are placeholders — not tuned,
    chapter3.tex marks the true dimensionality [TBD] pending ablation (its
    7-factor set differs from RMA's original 17)."""

    payload_mass_dim: int = 1
    payload_com_offset_dim: int = 3
    friction_dim: int = 1
    motor_power_scale_dim: int = 1
    leg_length_scale_dim: int = 1
    joint_range_scale_dim: int = 1
    terrain_height_dim: int = 1

    payload_treatment: Literal["explicit_observed_rewarded", "noise_only"] = "explicit_observed_rewarded"

    adaptation_latent_dim: int = 8  # z_t width — RMA's original default, [TBD] pending ablation

    @property
    def payload_dim(self) -> int:
        return self.payload_mass_dim + self.payload_com_offset_dim

    @property
    def dim(self) -> int:
        """Total e_t width — shrinks under noise_only (payload excluded
        entirely, not just unrewarded — see Pipeline_Summary.md §3.10)."""
        non_payload = (
            self.friction_dim + self.motor_power_scale_dim + self.leg_length_scale_dim
            + self.joint_range_scale_dim + self.terrain_height_dim
        )
        if self.payload_treatment == "noise_only":
            return non_payload
        return non_payload + self.payload_dim
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_extrinsics_cfg.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add talon_rl/config.py tests/test_extrinsics_cfg.py
git commit -m "feat: add ExtrinsicsCfg for Adaptation Module Phase 1"
```

---

## Task 2: `EnvFactorEncoder` module

**Files:**
- Create: `scripts/rl/core/modules/env_factor_encoder.py`
- Test: `tests/test_env_factor_encoder.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure `nn.Module`, no dependency on `ExtrinsicsCfg` — takes plain `int` dims so it stays testable in isolation).
- Produces: `EnvFactorEncoder(extrinsics_dim: int, latent_dim: int, hidden_dim: int = 32)` with `.forward(e_t: torch.Tensor) -> torch.Tensor` mapping `(B, extrinsics_dim) -> (B, latent_dim)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_factor_encoder.py
import torch

from rl.core.modules.env_factor_encoder import EnvFactorEncoder


def test_forward_shape():
    encoder = EnvFactorEncoder(extrinsics_dim=9, latent_dim=8)
    e_t = torch.randn(4, 9)
    z_t = encoder(e_t)
    assert z_t.shape == (4, 8)


def test_gradients_flow_to_all_parameters():
    # Non-obvious requirement: mu must joint-train with the policy through
    # the SAME optimizer (RMA's architecture) — if a layer's gradient is
    # zero/None here, MOPPOTrainer's backward pass silently wouldn't train
    # it either.
    encoder = EnvFactorEncoder(extrinsics_dim=9, latent_dim=8)
    e_t = torch.randn(4, 9)
    z_t = encoder(e_t)
    z_t.sum().backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.any(param.grad != 0), f"{name} got an all-zero gradient"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_env_factor_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.core.modules.env_factor_encoder'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rl/core/modules/env_factor_encoder.py
"""Env Factor Encoder (mu, RMA Phase 1) — compresses privileged extrinsics
e_t into a latent z_t. Lives here (trainer/algorithm layer), not in an
Isaac Lab Manager, because it must joint-train with the base policy
through the same optimizer (see scripts/rl/core/algorithms/moppo.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnvFactorEncoder(nn.Module):
    def __init__(self, extrinsics_dim: int, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(extrinsics_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        return self.net(e_t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_env_factor_encoder.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/rl/core/modules/env_factor_encoder.py tests/test_env_factor_encoder.py
git commit -m "feat: add EnvFactorEncoder module (RMA Phase 1 mu)"
```

---

## Task 3: `MOPPOTrainer` wiring (encoder, obs concat, checkpoint)

This task uses a **fake extrinsics-providing env** (matching how `tests/test_moppo_smoke.py` already tests `DummyTalonEnv` without Isaac Sim) so it's fully testable without GPU. Real Isaac Lab wiring is Task 6.

**Files:**
- Modify: `scripts/rl/core/algorithms/moppo.py`
- Modify: `tests/test_moppo_smoke.py`

**Interfaces:**
- Consumes: `EnvFactorEncoder` (Task 2), `ExtrinsicsCfg` (Task 1).
- Produces: `MOPPOTrainer.__init__(..., extrinsics_cfg: ExtrinsicsCfg | None = None)` — when given, builds `self.encoder`, adds its params to `self.optim`, and expects `transition["extrinsics"]` (shape `(N, extrinsics_cfg.dim)`) from every `env.step()`/`env.reset()` call; when `None` (the default — keeps `--env dummy` working unmodified), no encoder is built and no `extrinsics` key is required.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_moppo_smoke.py
from talon_rl.config import ExtrinsicsCfg


class _ExtrinsicsDummyEnv(DummyTalonEnv):
    """DummyTalonEnv plus a fake extrinsics vector in the transition dict —
    exercises MOPPOTrainer's encoder wiring without needing Isaac Sim."""

    def __init__(self, *args, extrinsics_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._extrinsics_dim = extrinsics_dim

    def _with_extrinsics(self, transition):
        transition["extrinsics"] = np.random.randn(self.num_envs, self._extrinsics_dim).astype(np.float32)
        return transition

    def reset(self):
        return self._with_extrinsics(super().reset())

    def step(self, action):
        transition, done = super().step(action)
        return self._with_extrinsics(transition), done


def test_encoder_concatenates_z_t_into_actor_obs():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    expected_actor_dim = trainer.stack.policy_obs_dim + reward_cfg.dim + extrinsics_cfg.adaptation_latent_dim
    assert trainer.model.actor_body[0].in_features == expected_actor_dim


def test_encoder_params_are_in_the_optimizer():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    optim_param_ids = {id(p) for group in trainer.optim.param_groups for p in group["params"]}
    for p in trainer.encoder.parameters():
        assert id(p) in optim_param_ids


def test_update_runs_end_to_end_with_encoder():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    stats = trainer.update()
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


def test_encoder_checkpoint_round_trips():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()
    moppo_cfg = MOPPOConfig(num_steps=5, epochs_per_update=1)

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, extrinsics_cfg=extrinsics_cfg, seed=0
    )
    trainer.update()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        trainer.save(ckpt_path)

        env2 = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=1, extrinsics_dim=extrinsics_cfg.dim)
        fresh_trainer = MOPPOTrainer(
            env2, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, extrinsics_cfg=extrinsics_cfg, seed=1
        )
        for p1, p2 in zip(trainer.encoder.parameters(), fresh_trainer.encoder.parameters()):
            assert not torch.equal(p1, p2)

        fresh_trainer.load(ckpt_path)
        for p1, p2 in zip(trainer.encoder.parameters(), fresh_trainer.encoder.parameters()):
            assert torch.equal(p1, p2)


def test_dummy_env_without_extrinsics_still_works():
    # --env dummy must keep working unmodified when extrinsics_cfg is None.
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0
    )
    assert not hasattr(trainer, "encoder") or trainer.encoder is None
    stats = trainer.update()
    assert np.isfinite(stats["policy_loss"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_moppo_smoke.py -v -k encoder`
Expected: FAIL — `MOPPOTrainer.__init__() got an unexpected keyword argument 'extrinsics_cfg'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/rl/core/algorithms/moppo.py`, add the import:

```python
from ..modules.env_factor_encoder import EnvFactorEncoder
```

Modify `MOPPOTrainer.__init__`'s signature (find the existing `def __init__(self, env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=None, stack_cfg=None, seed=0):`) to accept `extrinsics_cfg: ExtrinsicsCfg | None = None` (also import `ExtrinsicsCfg` from `talon_rl.config`), and right after `self.model = ActorCritic(...)` is constructed, insert:

```python
        self.extrinsics_cfg = extrinsics_cfg
        self.encoder = EnvFactorEncoder(extrinsics_cfg.dim, extrinsics_cfg.adaptation_latent_dim) if extrinsics_cfg else None
```

`actor_obs_w_dim`/`critic_obs_w_dim` (computed just before `self.model = ActorCritic(...)`) must add `extrinsics_cfg.adaptation_latent_dim` when `extrinsics_cfg` is set:

```python
        actor_obs_w_dim = self.stack.policy_obs_dim + reward_cfg.dim
        critic_obs_w_dim = self.stack.critic_obs_dim + reward_cfg.dim
        if extrinsics_cfg:
            actor_obs_w_dim += extrinsics_cfg.adaptation_latent_dim
            critic_obs_w_dim += extrinsics_cfg.adaptation_latent_dim
```

`self.optim` construction changes from `torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)` to:

```python
        params = list(self.model.parameters()) + (list(self.encoder.parameters()) if self.encoder else [])
        self.optim = torch.optim.Adam(params, lr=self.cfg.lr)
```

In `_collect_rollout`, after `transition, done = self.env.step(action)` and the existing `self.stack.push(...)` line, insert the encoder step and change the `actor_obs_w`/`critic_obs_w` concatenation lines (there are two concatenation sites in this method — the one right after `env.step()` and the one used for `act_inference`/initial construction; update the one inside the per-step loop first):

```python
            if self.encoder:
                z_t = self.encoder(torch.from_numpy(transition["extrinsics"]).float()).detach().numpy()
                actor_obs_w = np.concatenate([self.stack.policy_obs, self.w, z_t], axis=-1).astype(np.float32)
                critic_obs_w = np.concatenate([self.stack.critic_obs, self.w, z_t], axis=-1).astype(np.float32)
```

placed to replace the existing pre-step `actor_obs_w = np.concatenate([self.stack.policy_obs, self.w], axis=-1)...` computation — note this must be computed from the **current** `self.stack` state used to select the action for this step, so keep the existing pre-step obs (used for `self.model.act`) unchanged, and add the encoder step only where `z_t` is available (i.e. any per-step `actor_obs_w`/`critic_obs_w` construction needs the *current* extrinsics; since extrinsics for step `t`'s action-selection come from the transition *entering* step `t`, which is available from the previous step's `transition` or the initial `env.reset()` — store `self._last_extrinsics` similarly to how `self.stack` already carries forward `policy_obs`/`critic_obs` across steps). Concretely:

- In `__init__`, after `transition = self.env.reset()`, add: `self._last_extrinsics = transition.get("extrinsics") if self.encoder else None`
- Everywhere `actor_obs_w = np.concatenate([self.stack.policy_obs, self.w], axis=-1)` currently appears (there are three call sites: `_collect_rollout`'s per-step action-selection, `act_inference`, and the final bootstrap value computation for `critic_obs_w`), change to a small helper method to avoid triplicating the encoder logic:

```python
    def _actor_obs(self) -> np.ndarray:
        parts = [self.stack.policy_obs, self.w]
        if self.encoder:
            z_t = self.encoder(torch.from_numpy(self._last_extrinsics).float()).detach().numpy()
            parts.append(z_t)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _critic_obs(self) -> np.ndarray:
        parts = [self.stack.critic_obs, self.w]
        if self.encoder:
            z_t = self.encoder(torch.from_numpy(self._last_extrinsics).float()).detach().numpy()
            parts.append(z_t)
        return np.concatenate(parts, axis=-1).astype(np.float32)
```

Replace all three `np.concatenate([self.stack.policy_obs, self.w], axis=-1).astype(np.float32)` call sites with `self._actor_obs()`, and the corresponding critic ones with `self._critic_obs()`. After `transition, done = self.env.step(action)` in `_collect_rollout`, add `self._last_extrinsics = transition.get("extrinsics") if self.encoder else None` right next to the existing `self.stack.push(transition["obs"], done_mask=done)` line.

Finally, `save`/`load` gain the encoder's state (inside the existing `torch.save({...})`/`checkpoint = torch.load(...)` dicts):

```python
                "encoder": self.encoder.state_dict() if self.encoder else None,
```

and in `load`:

```python
        if self.encoder and checkpoint.get("encoder"):
            self.encoder.load_state_dict(checkpoint["encoder"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_moppo_smoke.py -v`
Expected: PASS (all tests, including the 5 new ones and every pre-existing test — `_actor_obs`/`_critic_obs` refactor must not change behavior when `encoder is None`)

- [ ] **Step 5: Commit**

```bash
git add scripts/rl/core/algorithms/moppo.py tests/test_moppo_smoke.py
git commit -m "feat: wire EnvFactorEncoder into MOPPOTrainer (z_t into actor/critic obs)"
```

---

## Task 4: Custom `randomize_joint_range` event function

**Files:**
- Create: `talon_rl/tasks/locomotion/a1_env/mdp/events.py`
- Modify: `talon_rl/tasks/locomotion/a1_env/mdp/__init__.py`

This needs a real `Articulation` and physics view — no CPU-only unit test is possible; verification happens in Task 6's GPU-gated test. Write it now per the verified `Articulation.write_joint_position_limit_to_sim` API (`isaaclab/assets/articulation/articulation.py:666`) and the `_randomize_prop_by_op`-style pattern the stock event functions already use.

**Interfaces:**
- Produces: `randomize_joint_range(env, env_ids, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg)` — an `EventTermCfg`-compatible function.

- [ ] **Step 1: Write the function**

```python
# talon_rl/tasks/locomotion/a1_env/mdp/events.py
"""Custom domain-randomization event functions this A1 task needs beyond
Isaac Lab's stock isaaclab.envs.mdp.events — see
docs/superpowers/specs/2026-09-15-adaptation-module-phase1-design.md.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def randomize_joint_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scales each env's joint position limits by a per-env factor around
    the default limits' midpoint (extrinsics factor: joint angle range,
    chapter3.tex §3.2.1). Written via Articulation.write_joint_position_
    limit_to_sim — a plain property write on the already-spawned
    Articulation, unlike leg-length (a geometry change Isaac Lab blocks
    for Articulations at runtime — see the spec)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    default_limits = asset.data.default_joint_pos_limits[env_ids]  # (E, J, 2)
    mean = default_limits.mean(dim=-1)  # (E, J)
    half_range = (default_limits[..., 1] - default_limits[..., 0]) / 2  # (E, J)

    scale = math_utils.sample_uniform(
        scale_range[0], scale_range[1], (len(env_ids), 1), device=asset.device
    )  # (E, 1), broadcasts over J

    new_half_range = half_range * scale
    new_limits = torch.stack([mean - new_half_range, mean + new_half_range], dim=-1)  # (E, J, 2)

    asset.write_joint_position_limit_to_sim(new_limits, env_ids=env_ids)
```

- [ ] **Step 2: Re-export from the task's `mdp` package**

```python
# talon_rl/tasks/locomotion/a1_env/mdp/__init__.py — add this line
from .events import randomize_joint_range  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
git add talon_rl/tasks/locomotion/a1_env/mdp/events.py talon_rl/tasks/locomotion/a1_env/mdp/__init__.py
git commit -m "feat: add randomize_joint_range event (Isaac Lab can't scale-randomize an Articulation for leg-length, but joint limits are a plain property write)"
```

(Real verification happens in Task 6's GPU-gated test — this task alone has no CPU-testable unit, matching how `tests/test_a1_env.py` already handles Isaac-Sim-only code.)

---

## Task 5: Offline leg-length USD-variant generation script

**Files:**
- Create: `scripts/rl/assets/generate_a1_leg_length_variants.py`

This is a standalone utility run once ahead of training (not part of the training loop, not imported by `train_prelim.py`) — needs a GPU machine with Isaac Sim installed to run (USD stage APIs require Kit's runtime, same constraint as everything else Isaac-Sim-touching in this repo). No unit test — it produces files, verified by inspecting the generated USDs exist and load (Task 6).

**Interfaces:**
- Produces: N files `talon_rl/assets/data/Robots/unitree_a1/unitree_a1_leg_scale_<s>.usd` (alongside the existing vendored `a1.usd`), each with a custom `legScale` attribute on the root prim readable back at runtime.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/rl/assets/generate_a1_leg_length_variants.py
"""Offline utility — generates N leg-length USD variants of the vendored
A1 asset for Isaac Lab's MultiUsdFileCfg spawn-time selection (leg-length
is fixed per env at spawn, not randomized at reset — see
docs/superpowers/specs/2026-09-15-adaptation-module-phase1-design.md for
why Isaac Lab blocks runtime scale-randomization on an Articulation).

Run once, on a machine with Isaac Sim installed:
    python scripts/rl/assets/generate_a1_leg_length_variants.py

For each scale factor s, every leg link (thigh, calf — hip stays fixed,
it's the mount point) gets:
  - geometry scaled by s (xformOp:scale)
  - mass scaled by s^3, diagonal inertia scaled by s^5 (uniform-density
    assumption, standard geometric similarity — NOT ported from any
    specific paper, see the spec's corrected note on the URMA citation)
  - the child joint's local position offset scaled by s (keeps the
    kinematic chain attached at the right point)
  - the chosen scale factor written as a custom "legScale" float attribute
    on the root prim, so an observation function can read back which
    variant a given env got (Isaac Lab doesn't expose "which multi-asset
    variant this env received" as a queryable scene property)
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402 — after SimulationApp, see repo convention

SCALE_FACTORS = [0.85, 0.925, 1.0, 1.075, 1.15]  # [TBD] placeholder range, not tuned — see spec's Open Questions
LEG_LINK_NAMES = [
    f"{side}_{seg}" for side in ("FR", "FL", "RR", "RL") for seg in ("thigh", "calf")
]

BASE_USD = "talon_rl/assets/data/Robots/unitree_a1/unitree_a1.usd"
OUT_DIR = "talon_rl/assets/data/Robots/unitree_a1"


def generate_variant(scale: float) -> str:
    out_path = os.path.join(OUT_DIR, f"unitree_a1_leg_scale_{scale}.usd")
    stage = Usd.Stage.Open(BASE_USD)
    stage.Export(out_path)  # start from a copy, edit the copy
    stage = Usd.Stage.Open(out_path)

    root_prim = stage.GetDefaultPrim()
    legscale_attr = root_prim.CreateAttribute("legScale", Sdf.ValueTypeNames.Float)
    legscale_attr.Set(scale)

    for link_name in LEG_LINK_NAMES:
        for prim in stage.Traverse():
            if prim.GetName() != link_name:
                continue

            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddScaleOp().Set((scale, scale, scale))

            if not UsdPhysics.MassAPI(prim):
                UsdPhysics.MassAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI(prim)
            current_mass = mass_api.GetMassAttr().Get() or 1.0
            current_inertia = mass_api.GetDiagonalInertiaAttr().Get()
            mass_api.CreateMassAttr().Set(current_mass * scale**3)
            if current_inertia is not None:
                mass_api.CreateDiagonalInertiaAttr().Set(tuple(v * scale**5 for v in current_inertia))

            for child in prim.GetChildren():
                joint = UsdPhysics.Joint(child)
                if not joint:
                    continue
                local_pos1 = joint.GetLocalPos1Attr().Get()
                if local_pos1 is not None:
                    joint.CreateLocalPos1Attr().Set(tuple(v * scale for v in local_pos1))

    stage.GetRootLayer().Save()
    return out_path


def main() -> None:
    paths = [generate_variant(s) for s in SCALE_FACTORS]
    for p in paths:
        print(f"generated: {p}")
    simulation_app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the GPU machine, verify files exist**

Run: `source ~/isaac-lab-env/bin/activate && PYTHONPATH=. python scripts/rl/assets/generate_a1_leg_length_variants.py`
Expected: 5 lines of `generated: talon_rl/assets/data/Robots/unitree_a1/unitree_a1_leg_scale_<s>.usd`, and `ls talon_rl/assets/data/Robots/unitree_a1/` shows all 5 new files.

- [ ] **Step 3: Commit the script (not the generated USD binaries — those regenerate from the script; add the output dir pattern to `.gitignore` if the team decides not to commit binaries, otherwise commit them like the other vendored assets)**

```bash
git add scripts/rl/assets/generate_a1_leg_length_variants.py
git commit -m "feat: offline USD-variant generator for leg-length randomization"
```

---

## Task 6: `a1_env_cfg.py` wiring — `EventsCfg`, `PrivilegedCfg`, `MultiUsdFileCfg` spawn

**Files:**
- Modify: `talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py`
- Create: observation functions in `talon_rl/tasks/locomotion/a1_env/mdp/observations.py`
- Modify: `tests/test_a1_env.py` (GPU-gated)

This task is GPU-gated end to end (Isaac Lab scene/manager construction). Depends on Task 4 (`randomize_joint_range`) and Task 5 (USD variants must exist on disk).

**Interfaces:**
- Consumes: `randomize_joint_range` (Task 4), the 5 generated USD variant paths (Task 5).
- Produces: `a1_env_cfg.py`'s `A1SceneCfg.robot.spawn` using `MultiUsdFileCfg`; `ObservationsCfg.privileged: PrivilegedCfg`; `EventsCfg` with 4 randomization terms (payload mass, CoM, friction, motor power, joint range — 5 terms; leg-length is spawn-time, not an event).

- [ ] **Step 1: Add extrinsics observation functions**

```python
# append to talon_rl/tasks/locomotion/a1_env/mdp/observations.py
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def payload_extrinsics(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload mass + CoM offset on the trunk (extrinsics factor 1+2).
    Only wired into PrivilegedCfg when payload_treatment != noise_only —
    see a1_env_cfg.py."""
    asset: Articulation = env.scene[asset_cfg.name]
    trunk_id = asset.find_bodies("trunk")[0][0]
    mass = asset.root_physx_view.get_masses()[:, trunk_id : trunk_id + 1]
    com = asset.root_physx_view.get_coms()[:, trunk_id, :3]
    return torch.cat([mass, com], dim=-1)


def friction_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    return materials[:, 0, 0:1]  # static friction of the first shape, as a per-env scalar


def motor_power_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    stiffness = next(iter(asset.actuators.values())).stiffness
    return stiffness.mean(dim=-1, keepdim=True)


def leg_length_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reads back the legScale custom attribute Task 5's generator script
    wrote onto each spawned variant's root prim."""
    asset: Articulation = env.scene[asset_cfg.name]
    scales = []
    for i in range(env.num_envs):
        prim = asset._root_physx_view.prim_paths[i]  # noqa: SLF001 — no public per-env prim accessor
        attr = env.scene.stage.GetPrimAtPath(prim).GetAttribute("legScale")
        scales.append(attr.Get() if attr.IsValid() else 1.0)
    return torch.tensor(scales, device=asset.device).unsqueeze(-1)


def joint_range_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits
    ranges = (limits[..., 1] - limits[..., 0]).mean(dim=-1, keepdim=True)
    default_ranges = (asset.data.default_joint_pos_limits[..., 1] - asset.data.default_joint_pos_limits[..., 0]).mean(dim=-1, keepdim=True)
    return ranges / default_ranges  # current/default ratio — this env's scale factor


def local_terrain_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terrain height needs no new randomization — it's already
    effectively randomized by which sub-terrain cell a lane spawns on
    (A1_ROUGH_TERRAINS_CFG). Reads the terrain's own tracked level per env."""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels.float() if hasattr(terrain, "terrain_levels") else torch.zeros(env.num_envs, device=env.device)
    return levels.unsqueeze(-1)
```

- [ ] **Step 2: Wire `EventsCfg`, `ObservationsCfg.PrivilegedCfg`, and the `MultiUsdFileCfg` spawn**

In `talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py`:

```python
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
```

Replace `A1SceneCfg.robot`'s existing single-`UsdFileCfg` spawn (find where `TALON_A1_CFG` is used/assigned to `robot`) so its `spawn` field becomes:

```python
    robot: ArticulationCfg = TALON_A1_CFG.replace(
        spawn=MultiUsdFileCfg(
            usd_path=[
                f"talon_rl/assets/data/Robots/unitree_a1/unitree_a1_leg_scale_{s}.usd"
                for s in (0.85, 0.925, 1.0, 1.075, 1.15)
            ],
            random_choice=True,
        )
    )
```

Add a new `EventsCfg` class (after `TerminationsCfg`):

```python
@configclass
class EventsCfg:
    randomize_payload_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (0.0, 5.0),  # kg, matches Pipeline_Summary.md's Pareto-sweep table
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_payload_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},  # [TBD] placeholder, not tuned
        },
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 1.2),  # [TBD] placeholder, not tuned
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    randomize_motor_power = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),  # [TBD] placeholder, not tuned
            "operation": "scale",
        },
    )
    randomize_joint_range = EventTerm(
        func=mdp.randomize_joint_range,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.8, 1.0)},  # [TBD] placeholder, not tuned
    )
```

Extend `ObservationsCfg` (the `payload` term is conditionally added in `__post_init__` — matches `RewardVectorCfg.active`'s existing toggle pattern):

```python
@configclass
class ObservationsCfg:
    class PolicyCfg(ObsGroup):
        ...  # UNCHANGED — do not add any extrinsics term here

    @configclass
    class PrivilegedCfg(ObsGroup):
        friction = ObsTerm(func=mdp.friction_extrinsic)
        motor_power = ObsTerm(func=mdp.motor_power_extrinsic)
        leg_length = ObsTerm(func=mdp.leg_length_extrinsic)
        joint_range = ObsTerm(func=mdp.joint_range_extrinsic)
        terrain_height = ObsTerm(func=mdp.local_terrain_height)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

    def __post_init__(self):
        if self.extrinsics_cfg.payload_treatment != "noise_only":
            self.privileged.payload = ObsTerm(func=mdp.payload_extrinsics)
```

(`ObservationsCfg` needs an `extrinsics_cfg: ExtrinsicsCfg` field threaded in from `IsaacLabTalonEnvCfg`'s own `__post_init__`, matching how `cfg.scene.num_envs` is already set externally after construction — add `extrinsics_cfg: ExtrinsicsCfg = field(default_factory=ExtrinsicsCfg)` to `ObservationsCfg` and call `self.observations.__post_init__()` again, or simplest: have `IsaacLabTalonEnvCfg.__post_init__` conditionally do `self.observations.privileged.payload = ObsTerm(...)` directly instead of nesting the logic inside `ObservationsCfg` — follow whichever pattern `cfg.scene.num_envs = args.num_envs`'s existing post-construction mutation in `train_prelim.py` already establishes, to stay consistent with this repo's convention rather than fighting `@configclass`'s dataclass-like semantics.)

Add `events: EventsCfg = EventsCfg()` to `IsaacLabTalonEnvCfg` alongside its existing `terminations`/`observations` fields.

- [ ] **Step 3: Extend the GPU-gated structural test**

```python
# add to tests/test_a1_env.py, inside test_isaac_lab_env_implements_base_contract's try block,
# after the existing terrain assertion
        assert "privileged" in env.scene... # placeholder shape — replace with the actual ObservationManager
        # group-introspection call once run on the GPU machine and the exact
        # API is confirmed (env.observation_manager.active_terms or similar,
        # per isaaclab.managers.observation_manager's actual introspection
        # surface — verify against installed isaaclab 0.48.0 same as every
        # other assertion in this file).
```

(Leave the exact introspection call for the executor to fill in against the real installed API when running this on the GPU machine — every other assertion in this file was written the same way, verified live rather than guessed; this is the one step in this plan that's intentionally GPU-verify-at-execution-time rather than pre-written, matching this file's own established pattern.)

- [ ] **Step 4: Run on the GPU machine**

Run: `source ~/isaac-lab-env/bin/activate && PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_a1_env.py -v -s`
Expected: PASS, `Observation Manager` banner shows both `policy` and `privileged` groups, `Event Manager` banner shows 5 active terms.

- [ ] **Step 5: Commit**

```bash
git add talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py talon_rl/tasks/locomotion/a1_env/mdp/observations.py tests/test_a1_env.py
git commit -m "feat: wire Adaptation Module Phase 1 extrinsics into A1 env (EventsCfg, PrivilegedCfg, MultiUsdFileCfg spawn)"
```

---

## Task 7: `train_prelim.py` — pass `extrinsics_cfg` through for `--env isaac_lab`

**Files:**
- Modify: `scripts/rl/train_prelim.py`

**Interfaces:**
- Consumes: `ExtrinsicsCfg` (Task 1), `MOPPOTrainer`'s new `extrinsics_cfg` param (Task 3).

- [ ] **Step 1: Wire it in**

In `main()`, after `obs_cfg = ObservationSpaceCfg()` and friends are constructed, add:

```python
    extrinsics_cfg = ExtrinsicsCfg() if args.env == "isaac_lab" else None
```

Pass `extrinsics_cfg=extrinsics_cfg` into the `MOPPOTrainer(...)` construction call. Add the import: `from talon_rl.config import ExtrinsicsCfg` alongside the existing `talon_rl.config` import line.

- [ ] **Step 2: Run the existing dummy-env smoke command to confirm nothing broke**

Run: `source .venv/bin/activate && PYTHONPATH=".:scripts" python scripts/rl/train_prelim.py --env dummy --updates 2 --num_envs 4`
Expected: runs identically to before (no encoder — `extrinsics_cfg=None` for `--env dummy`).

- [ ] **Step 3: Commit**

```bash
git add scripts/rl/train_prelim.py
git commit -m "feat: pass ExtrinsicsCfg through train_prelim.py for --env isaac_lab"
```

---

## Self-Review Notes

1. **Spec coverage:** All 7 extrinsics (Task 6's `EventsCfg`/`observations.py`), leg-length's corrected offline-generation approach (Task 5), `PrivilegedCfg` separate from `PolicyCfg` (Task 6), encoder in the trainer layer (Task 2-3), `payload_treatment` shrinking $e_t$'s width (Task 1, Task 6's conditional `payload` term), checkpointing (Task 3), `--env dummy` untouched (Task 3's `test_dummy_env_without_extrinsics_still_works`, Task 7). Payload reward term formula and most randomization ranges are explicitly out of scope per the spec — not silently dropped, called out inline as `[TBD] placeholder` at every params dict in Task 6.
2. **Placeholder scan:** Task 6 Step 3's observation-manager introspection call is the one intentional exception — flagged explicitly as GPU-verify-at-execution-time, matching `tests/test_a1_env.py`'s own established pattern (every assertion in that file was written after live verification, not before). Every other step has real, complete code.
3. **Type consistency:** `ExtrinsicsCfg.dim`/`adaptation_latent_dim` (Task 1) match `EnvFactorEncoder(extrinsics_dim, latent_dim)`'s constructor args (Task 2) match `MOPPOTrainer`'s usage (Task 3) match `train_prelim.py`'s construction (Task 7). `randomize_joint_range`'s signature (Task 4) matches its `EventTerm(func=mdp.randomize_joint_range, params={...})` call site (Task 6).
