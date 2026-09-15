# Adaptation Module Phase 1 — Env Factor Encoder (design)

## Context

chapter3.tex §3.2.1 specifies a two-phase teacher-student Adaptation Module
(RMA, Kumar et al. 2021), tagged `[INFRA]` (infrastructure borrowed from
validated literature, not this thesis's core contribution — the
Multi-Objective Module is). Phase 1 (teacher, privileged): an Env Factor
Encoder μ compresses 7 privileged environment/morphology factors — mass,
center of gravity, friction, terrain height, motor power, leg length, joint
angle range — into a latent $z_t$, trained jointly with the base policy.
Phase 2 (student distillation): a separate Adaptation Module φ learns to
estimate $\hat z_t, \sigma_t$ from proprioceptive history alone (no
privileged access), regressed against $z_t$ from Phase 1's rollouts.

This spec covers **Phase 1 only**. See
`03_Daily_Notes/2026-09-15.md` (talon-thesis) for the fuller brainstorm
history — extrinsics dimensioning, the leg-length physics-consistency
problem and its fix, and why this isn't vendored as a new Isaac Lab Manager.

Also relevant, from `00_Proposal/Pipeline/Pipeline_Summary.md` §3.10 (the
4-way experiment design): a `payload_treatment` config flag
(`explicit_observed_rewarded` for Main vs. `noise_only` for Baseline A)
governs whether payload mass/CoM specifically are part of $e_t$ at all, and
whether they drive a separate reward term. This flag's exact reward formula
is an **open question** (§Open Questions below) — everything else in this
spec is buildable and testable without it resolved.

## Scope

**In scope**:
- All 7 extrinsics factors in $e_t$ (no factors deferred to a later spec)
- Isaac Lab domain randomization for each factor via `EventManager` (stock
  functions where they exist; new custom event functions for leg-length
  scaling and joint-range scaling)
- A new `Privileged` observation group (via `ObservationManager`'s existing
  multiple-group support) exposing raw $e_t$ — separate from the policy's
  existing `PolicyCfg` group, which must **not** gain these terms (Phase 2's
  whole premise is that the deployed policy never sees privileged state)
- `EnvFactorEncoder` (μ, a small trainable `nn.Module`) in
  `scripts/rl/core/modules/`, joint-trained with the base policy via
  `MOPPOTrainer`'s existing optimizer
- `MOPPOTrainer` wiring: pull $e_t$ from the env, encode to $z_t$,
  concatenate into actor/critic obs (same mechanical pattern the preference
  vector $w$ already uses), checkpoint the encoder's state
- `payload_treatment` config flag, threaded through $e_t$'s composition
  (not just a reward toggle — see Data Flow)

**Out of scope** (follow-up specs):
- Phase 2 (Adaptation Module φ, distillation training loop, `t̂_z, σ_t`)
- Gate 1 (iterative domain-randomization-range testing) and Gate 2
  (distillation error threshold, dilated-conv-vs-CNN ablation)
- HLP (High-Level Policy)
- `preference_architecture`'s other conditions (Baseline B/C) — those are
  Multi-Objective Module concerns, already implemented for D3PO (`Main`) vs.
  AMOR (`Baseline C`) as of `moppo.py`'s 2026-09-15 D3PO LSW change; this
  spec only threads `payload_treatment` through
- Resolving the payload reward term's exact formula (§Open Questions)

## Why not a new vendored Manager

Considered and rejected: a bespoke `AdaptationManager` (in the style of the
already-vendored `ConstraintManager`,
`docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md`).
Rejected because:

1. **Domain randomization of 6 of the 7 factors is exactly what Isaac
   Lab's stock `EventManager` already does** — most have ready-made
   functions in `isaaclab.envs.mdp.events` (mass with `recompute_inertia`,
   friction, actuator gain); joint-range needs one new event *function*,
   not a new manager *type*. Leg-length is the exception — it's not an
   event at all (Isaac Lab blocks runtime scale-randomization on an
   Articulation), it's an offline USD-variant-generation script plus stock
   `MultiUsdFileCfg` spawn-time selection (see below) — still no new
   manager, just a different stock mechanism.
2. **Exposing raw $e_t$ to the trainer is exactly what `ObservationManager`
   already does** — `a1_env_cfg.py` already has one obs group (`PolicyCfg`);
   Isaac Lab natively supports more than one, which is exactly the
   teacher/student split this module needs (privileged group now, policy
   group unchanged, Phase 2 adds nothing new to either).
3. **The encoder μ is a trainable `nn.Module` and must stay in the trainer
   layer, not an env-side Manager, in either design** — RMA jointly trains
   the encoder with the policy through one optimizer; Isaac Lab Managers
   don't hold gradient-tracked parameters or participate in a training
   loop's backward pass. Putting μ inside a Manager would sever it from
   `MOPPOTrainer.optim`, which is the one part of this design that isn't a
   choice.

A custom Manager earns its cost when there are multiple interchangeable,
composable terms with their own lifecycle (`ConstraintManager`'s per-term
curriculum ramping is a real example). Phase 1 has one encoder and, at most,
one new reward term — plain config on stock managers is the right size.

## Components

### `talon_rl/config.py`

```python
@dataclass(frozen=True)
class ExtrinsicsCfg:
    """Phase 1 privileged extrinsics e_t. Dimensions are placeholders — not
    tuned, chapter3.tex marks the true dimensionality [TBD] pending
    ablation (its 7-factor set differs from RMA's original 17)."""
    payload_mass_dim: int = 1
    payload_com_offset_dim: int = 3
    friction_dim: int = 1
    motor_power_scale_dim: int = 1
    leg_length_scale_dim: int = 1
    joint_range_scale_dim: int = 1
    terrain_height_dim: int = 1

    payload_treatment: Literal["explicit_observed_rewarded", "noise_only"] = "explicit_observed_rewarded"

    @property
    def payload_dim(self) -> int:
        return self.payload_mass_dim + self.payload_com_offset_dim

    @property
    def dim(self) -> int:
        """Total e_t width — shrinks under noise_only (payload excluded).
        See Data Flow: this is why payload_treatment lives on e_t's
        composition, not as a downstream reward-only toggle."""
        non_payload = (
            self.friction_dim + self.motor_power_scale_dim + self.leg_length_scale_dim
            + self.joint_range_scale_dim + self.terrain_height_dim
        )
        if self.payload_treatment == "noise_only":
            return non_payload
        return non_payload + self.payload_dim

    adaptation_latent_dim: int = 8  # z_t width — RMA's original default, [TBD] pending ablation
```

### Leg-length: offline USD-variant generation, not a runtime event

**Corrected 2026-09-15** (see `03_Daily_Notes/2026-09-15.md`, talon-thesis):
`isaaclab.envs.mdp.events.randomize_rigid_body_scale` explicitly raises
`ValueError` for an `Articulation` (A1 is one) — verified against the
installed isaaclab 0.48.0 source, not assumed. Isaac Lab's own docstring
recommends generating separate USD files and using multi-asset spawning
instead, which is what this spec does:

- `scripts/rl/assets/generate_a1_leg_length_variants.py` (new, offline
  utility — not part of the training loop, run once ahead of time): for
  each of $N$ discrete scale factors, opens the base A1 USD, scales the
  leg-link geometry, analytically recomputes mass ($\propto s^3$) and
  inertia ($\propto s^5$, uniform-density assumption) via
  `UsdPhysics.MassAPI`, shifts each child joint's local transform by $s$ so
  the kinematic chain stays consistent, writes the chosen scale factor as a
  custom USD attribute on the root prim (so it can be read back as an
  observation — Isaac Lab doesn't expose "which multi-asset variant this
  env got" as a queryable property), and saves as a new
  `unitree_a1_leg_scale_<s>.usd` file next to the vendored asset.
- `A1SceneCfg.robot.spawn` changes from a single `UsdFileCfg` to
  `MultiUsdFileCfg(usd_path=[<N generated variant paths>], random_choice=True)`
  — Isaac Lab's own stock wrapper (`isaaclab.sim.spawners.wrappers`),
  picks one variant per env at scene construction, not at each reset (leg
  length doesn't change mid-episode, unlike the other 6 factors).
- `mdp.leg_length_extrinsic` (observation function) reads the custom scale
  attribute back off each env's spawned prim.

### `talon_rl/tasks/locomotion/a1_env/mdp/events.py` (new file)

Custom event function Isaac Lab doesn't ship:
- `randomize_joint_range(env, asset_cfg, scale_range)` — scales each
  joint's position limits by a per-env factor, written via the
  Articulation's joint-limit runtime API. (Unlike leg-length, this is a
  property write on the already-spawned Articulation, not a geometry
  change — no scale-on-Articulation restriction applies.)

Existing stock functions reused directly (no new code): mass/CoM via
`isaaclab.envs.mdp.events.randomize_rigid_body_mass` (has a
`recompute_inertia` flag built in — verified in the installed source, no
custom mass/inertia code needed here, unlike leg-length's geometry case),
friction via `randomize_rigid_body_material`, motor power via
`randomize_actuator_gains`.

`terrain_height` needs no new event — it's already effectively randomized
by which sub-terrain cell a lane spawns on (`A1_ROUGH_TERRAINS_CFG`); Phase
1 only needs to *read* it back (next section).

### `a1_env_cfg.py`'s `EventsCfg` and `ObservationsCfg`

```python
@configclass
class EventsCfg:
    randomize_payload_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="reset", params={...})
    randomize_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="reset", params={...})
    randomize_motor_power = EventTerm(func=mdp.randomize_actuator_gains, mode="reset", params={...})
    randomize_joint_range = EventTerm(func=mdp.randomize_joint_range, mode="reset", params={...})
    # leg-length is NOT an event — it's fixed per env at spawn time via
    # A1SceneCfg.robot.spawn's MultiUsdFileCfg, see above.

@configclass
class ObservationsCfg:
    class PolicyCfg(ObsGroup):
        ...  # unchanged — no extrinsics added here, ever

    class PrivilegedCfg(ObsGroup):
        """Phase 1 only — Phase 2's whole premise is the deployed policy
        never sees this group."""
        payload = ObsTerm(func=mdp.payload_extrinsics)      # only present if payload_treatment != noise_only
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
```

`payload_treatment == "noise_only"` removes the `payload` term entirely
(handled by `ExtrinsicsCfg.__post_init__` or the env cfg's own
`__post_init__`, following the pattern `RewardVectorCfg.active` already
uses for toggling terms) — this is what makes $e_t$'s width itself
conditional, not just a downstream reward gate.

### `scripts/rl/core/modules/env_factor_encoder.py` (new file)

```python
class EnvFactorEncoder(nn.Module):
    """mu: e_t -> z_t. Small MLP, jointly trained with the base policy —
    see moppo.py for why this can't live in an Isaac Lab Manager."""
    def __init__(self, extrinsics_dim: int, latent_dim: int, hidden_dim: int = 32): ...
    def forward(self, e_t: torch.Tensor) -> torch.Tensor: ...
```

### `scripts/rl/core/algorithms/moppo.py`

- `__init__`: construct `self.encoder = EnvFactorEncoder(extrinsics_cfg.dim, extrinsics_cfg.adaptation_latent_dim)`;
  `self.optim` gains `self.encoder.parameters()` (joint training, per RMA).
- `_collect_rollout`: pull raw $e_t$ from the env's `privileged` obs group
  (transition dict gains an `extrinsics` key, same pattern as
  `v_actual`/`obstacle_dist`/etc. already there for reward computation);
  `z_t = self.encoder(e_t)`; concatenate onto `actor_obs_w`/`critic_obs_w`
  the same way $w$ already is.
- `save`/`load`: add `self.encoder.state_dict()` to the checkpoint,
  following the same reasoning as the reward normalizer (resuming with a
  freshly-initialized encoder would be a discontinuity the value function
  wasn't trained against).

### `DummyTalonEnv` path

Per the earlier brainstorm decision, Phase 1 targets **Isaac Lab directly**,
not the dummy env — no randomization/exposure work happens in
`scripts/rl/core/dummy_env.py` under this spec. `DummyTalonEnv`'s
transition dict simply won't have an `extrinsics` key; `MOPPOTrainer` must
not hard-require it (only build/use the encoder path when the env actually
provides one), so `--env dummy` keeps working unmodified for fast
iteration on everything *except* the Adaptation Module itself.

## Data Flow

```
EventManager (reset)                    ObservationManager (privileged group)
  randomize e_t's 7 factors  ─────────→  read back raw e_t
                                              │
                                              ▼
                                    MOPPOTrainer._collect_rollout
                                    z_t = encoder(e_t)
                                              │
                                              ▼
                              concat([policy_obs, w])  concat([policy_obs, w, z_t])?
```

**Open question folded into this spec, not deferred**: does $z_t$ join the
*actor's* obs (so the deployed policy conditions on it — matching Phase 2's
eventual $\hat z_t$ substitution) or only the *critic's* (asymmetric
actor-critic, since Phase 2 hasn't replaced $z_t$ with $\hat z_t$ yet and a
policy trained on privileged $z_t$ can't be deployed as-is)? RMA's own
answer: the **actor** conditions on $z_t$ in Phase 1 (that's the whole point
— Phase 2 trains φ to replace $z_t$ with $\hat z_t$ as a drop-in for the
*same* actor input, not to retrain a new actor). This spec follows RMA:
`actor_obs_w` gains $z_t$; the critic can optionally see it too (harmless,
asymmetric actor-critic already gives the critic more than the actor sees).

## Testing

Matches this repo's existing pattern (`tests/test_running_norm.py`,
`tests/test_d3po_loss.py`): pure math gets a dedicated unit-test file
without Isaac Sim; the Isaac Lab wiring itself needs the GPU machine
(`tests/test_a1_env.py`'s existing skip-without-isaacsim pattern).

- `tests/test_env_factor_encoder.py` — shape correctness, gradient flow
  (encoder params actually get nonzero gradients from a backward pass)
- `tests/test_extrinsics_cfg.py` — `ExtrinsicsCfg.dim` shrinks correctly
  under `noise_only`; `payload_dim` matches the sum of its two sub-fields
- `tests/test_a1_env.py` extension (GPU-gated, like the existing terrain
  assertion) — `privileged` obs group exists and has the expected width;
  `PolicyCfg` unchanged (still excludes all extrinsics)
- `tests/test_moppo_smoke.py` extension — encoder checkpoint round-trips;
  `update()` stays finite with the encoder wired in

## Open Questions

1. **Payload reward term's exact formula** (`explicit_observed_rewarded`
   only). Pipeline_Summary.md says Main gets "a separate reward term" for
   payload but doesn't specify what it rewards — payload-tracking accuracy
   doesn't make sense in Phase 1 (the encoder has privileged access, there's
   nothing to track yet; that's Phase 2's `‖ẑ_t − z_t‖` job). Candidates:
   a stability/balance term scaled by current payload, or something else
   the thesis author needs to decide. **Decision deferred to the user** —
   Phase 1's config plumbing (`payload_treatment` toggling $e_t$'s
   composition) is buildable and testable without this resolved; the reward
   term itself is a stubbed no-op until specified.
2. **Exact event-function parameter ranges** (mass ±kg, friction range,
   motor power scale range, leg-length scale range, joint-range scale
   range) — chapter3.tex doesn't fix these, and they interact with Gate 1's
   iterative "expand randomization range, retest" methodology (out of this
   spec's scope). Placeholder ranges go in `ExtrinsicsCfg`/event params,
   clearly marked not-tuned, matching `RewardVectorCfg`'s existing
   convention.
