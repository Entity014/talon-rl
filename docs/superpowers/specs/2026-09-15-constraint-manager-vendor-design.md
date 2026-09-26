# Vendor `ConstraintManager` from Isaac-RL-Two-wheel-Legged-Bot (design)

## Context

`jaykorea/Isaac-RL-Two-wheel-Legged-Bot`'s `lab/flamingo/isaaclab/isaaclab/managers/`
extends Isaac Lab's core manager framework with a `ConstraintManager` — a
manager type Isaac Lab itself does not ship. Instead of a hard
terminate-on-violation or a hand-tuned reward penalty, each constraint term
reports a continuous violation degree in `[0, 1]`; terms marked
`time_out="constraint"` are converted into a **stochastic termination
probability** (scaled by a running max of past violations and a
curriculum-scheduled `p_max`), so enforcement is lax early in training and
strict later. See [[2026-09-14-vectorized-isaac-lab-env-design]] for how
`talon_rl`'s current env stack (`BaseTalonEnv`, `IsaacLabTalonEnv`) is
structured; this addition does not touch that stack.

talon-rl has no current constraint requirement and no env wired to consume
this — it's requested now so the mechanism is available when a future env
needs a safety constraint (e.g. a torque or tip-over limit) that plain
reward shaping can't handle cleanly. This is a deliberate exception to the
project's normal YAGNI stance, made explicitly by the user.

## Scope

**In scope**: vendor `ConstraintTermCfg` and `ConstraintManager` only, cleaned
up, importable, unit-testable without Isaac Sim installed (mocking
Isaac Lab's `ManagerBase`/`ManagerTermBase`).

**Out of scope**: `ManagerBasedConstraintRLEnv` (the env subclass that would
actually consume this manager), wiring it into `IsaacLabTalonEnv` or
`a1_env_cfg.py`, and any concrete constraint term functions for the A1. None
of that has a design yet — it happens when a real env needs it.

## File layout

```
talon_rl/isaaclab/
├── __init__.py
└── managers/
    ├── __init__.py            # re-exports ConstraintTermCfg, ConstraintManager
    ├── constraint_term_cfg.py
    └── constraint_manager.py
```

Single-level `isaaclab/` package, not the source project's nested
`isaaclab/isaaclab/`. That double-nesting exists there so
`lab.flamingo.isaaclab.isaaclab.envs` shadows the real `isaaclab.envs` for
multiple `entry_point=` gym registration strings across their task configs —
a need that doesn't exist here since nothing registers or imports this
module yet.

## Changes from the vendored source

1. **Translate every Korean comment to English** — repo convention is
   English comments explaining why, not what.
2. **Delete `dones` and `get_termination_probs` properties** — both marked
   `#* not using yet` in the source, i.e. already-known-dead code. Add them
   back if and when a consumer needs them.
3. **Fix a latent bug**: `_prepare_terms()` appends to
   `self._class_constraint_cfgs`, but `__init__` only ever defines
   `self._class_term_cfgs` — a name mismatch that would raise
   `AttributeError` the first time any constraint term's `func` is a
   `ManagerTermBase` instance. Never triggered upstream because nothing
   exercises this path yet. Rename consistently to `_class_term_cfgs`.
4. **Hyperparameters become constructor kwargs**, not hardcoded body
   assignments — resolves the vendored `# TODO should be set by the user
   argument (agent_cfg)` comment:
   - `tau: float = 0.95`
   - `min_p: float = 0.0`
   - `num_transitions_per_env: int = 24`
   - `max_iterations: int = 5000`
   - `static_curriculum_steps: int = 30000`

   `step_cur` stays a derived value computed from the two iteration counts
   in `__init__`, not a separate kwarg.
5. **`TYPE_CHECKING` import** changes from
   `lab.flamingo.isaaclab.isaaclab.envs.ManagerBasedConstraintRLEnv` (doesn't
   exist here) to `isaaclab.envs.ManagerBasedRLEnv` (the stock Isaac Lab
   class) — `compute()`'s only use of `self.env` is
   `self.env.common_step_counter`, which exists on the stock class. A
   comment notes this should become a TALON-specific env subclass's type if
   and when one exists.
6. Everything else (the `[0,1]` violation-degree contract, the
   `truncate`/`terminate`/`constraint` three-way `time_out` semantics, the
   EMA running-max + curriculum-scaled probability math in `compute()`) is
   preserved as-is — it's the actual mechanism being vendored, not
   incidental cleanup surface.

## Dependencies

Add `prettytable` to `pyproject.toml`'s `dependencies` (used by
`ConstraintManager.__str__`). `isaaclab.managers.manager_base` (`ManagerBase`,
`ManagerTermBase`) and `isaaclab.utils.configclass` remain hard imports —
this module requires Isaac Sim/Isaac Lab to import at all, same as
`talon_rl/tasks/locomotion/a1_env/a1_env.py`.

## Testing

`isaaclab` isn't installed in the default dev environment (same situation as
`tests/tasks/a1/test_environment.py`, which is GPU/Isaac-Sim-gated and skips on the
3.12 `.venv`). New tests live at `tests/isaaclab/test_constraint_manager.py`,
skipped via the same `isaaclab`-import-guard pattern as `test_a1_env.py`
when Isaac Sim isn't present, plus a mock-based test path that stubs
`isaaclab.managers.manager_base.ManagerBase`/`ManagerTermBase` well enough to
exercise `ConstraintManager.compute()`'s three `time_out` branches
(`truncate`, `terminate`, `constraint`) and the curriculum math directly,
without a real Isaac Sim install. Covers at minimum:
- a `"constraint"` term's probability output stays in `[0, 1]` and respects
  `p_max`
- `use_curriculum=True` forces `p_max=0.0` before
  `static_curriculum_steps` and ramps afterward
- the `_class_term_cfgs` bugfix (a `ManagerTermBase`-typed `func` doesn't
  raise)
- `"truncate"`/`"terminate"` terms reject non-0/1 values as documented
