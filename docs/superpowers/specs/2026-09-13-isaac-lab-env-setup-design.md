# Isaac Lab environment setup — design

Date: 2026-09-13
Status: approved, pending implementation plan

## Context

README's "Next milestones" item 1 is: *"Write `IsaacLabTalonEnv(BaseTalonEnv)`
against the real Unitree A1 asset."* This repo currently only has
`DummyTalonEnv`, a physics-free smoke test — there is no GPU/Isaac Sim in the
environment it was scaffolded in. This design covers moving the prelim onto
a real GPU machine and standing up the first real (non-dummy) environment.

Target machine (checked directly, 2026-09-13):

- Local Ubuntu 24.04.5 LTS ("noble")
- NVIDIA GeForce RTX 3070 Ti, 8GB VRAM, driver 610.43.02
- 32GB RAM, 328GB free disk
- GLIBC 2.39
- `uv` 0.11.15 already installed
- Existing `talon-rl/.venv` is Python 3.12, used for the current dummy-env
  pytest suite

## Relationship to the D³PO/AMOR Main decision (2026-09-13, same day)

Separately, `chapter3.tex` §3.2.3/§3.10 now names **Main** as AMOR's architecture
with D³PO's Late-Stage Weighting loss (not AMOR's original early-scalarization,
which moved to `Baseline C`) — see `talon-thesis/03_Daily_Notes/2026-09-13.md`.
That decision is **orthogonal to this design**: `IsaacLabTalonEnv` only has to
satisfy `BaseTalonEnv`'s `reset()`/`step()` contract, and `training/moppo.py`
already doesn't need to change to swap envs (see §2 below) — the same holds in
reverse, swapping *which loss `moppo.py` runs* doesn't touch the env. Concretely:

- This design's smoke-test bar ("a handful of PPO updates run headless without
  crashing," §3) is satisfied by whichever `moppo.py` is checked in at the
  time — today that's still the AMOR-style scalarizer. There is no need to
  block Isaac Lab env work on D³PO's Late-Stage Weighting landing first.
- The env work here and the D³PO implementation in `moppo.py` (tracked as an
  open item in `talon-thesis/03_Daily_Notes/PROGRESS.md`, Stage 3) can proceed
  in either order or in parallel.
- What *does* depend on order: before the Isaac Lab smoke run is reported as
  testing §3.10's `Main` condition specifically (as opposed to just "the
  pipeline doesn't crash on a real sim"), `moppo.py` needs D³PO's Late-Stage
  Weighting implemented — otherwise the run is exercising `Baseline C`'s loss
  (AMOR early-scalarization), not `Main`, regardless of what the env is.

## Goal

Get a real Isaac Lab environment (`IsaacLabTalonEnv`) running the same
`obs → policy(w) → action → reward vector → vector critic → PPO update` loop
that `DummyTalonEnv` already proves out, on this machine, without breaking
anything the dummy env currently does.

## Non-goals

- Any locomotion result claim. Per README's existing rule, "the policy
  improved" is not a valid claim until there's a real terrain/env — this
  design only gets the pipeline running against a real physics sim, same
  bar as the dummy env's "doesn't crash."
- Adaptation Module, Exteroception Module, terrain curriculum, per-objective
  reward normalization — all explicitly out of scope per README, unaffected
  by this change.
- Multi-env vectorized collection for throughput — noted as a known gap,
  not solved here (see Risks below).

## 1. Environment setup (external tooling, not committed to this repo)

- Isaac Sim **5.X** via pip package, Isaac Lab pinned to tag **`release/2.3.0`**
  (last non-beta release line; the newer `3.0.0-beta2` line pins Isaac Sim
  6.0/6.1 but is still beta — stability was the explicit ask here).
- Separate venv, **outside this repo**: `~/isaac-lab-env`, created with
  `uv venv ~/isaac-lab-env --python 3.11` — Isaac Sim 5.X requires Python
  3.11 exactly; a mismatch is a hard error, not a warning. This is
  deliberately not the same venv as `talon-rl/.venv` (3.12, used for the
  dummy-env pytest suite) so neither install can break the other.
- Steps:
  1. `uv venv ~/isaac-lab-env --python 3.11`
  2. `uv pip install isaacsim[all]` (pip method — confirmed compatible with
     Ubuntu 24.04 / GLIBC 2.39; the binary/standalone installer is the
     fallback only if pip fails, and explicitly cannot be combined with a
     uv/venv-managed Python)
  3. `git clone` IsaacLab at tag `release/2.3.0` next to (not inside)
     `talon-rl/`
  4. `./isaaclab.sh --install` from inside that clone, using the
     `~/isaac-lab-env` venv
  5. Install `talon-rl` itself into the same venv (`uv pip install -e
     /path/to/talon-rl`) so `IsaacLabTalonEnv` can `import talon_rl`
- Verify install with a trivial headless launch
  (`SimulationApp({"headless": True})` + immediate shutdown) before writing
  any TALON-specific code — isolates "Isaac Sim installed correctly" from
  "our env code is correct."

## 2. `IsaacLabTalonEnv(BaseTalonEnv)`

New file: `talon_rl/envs/isaac_lab_env.py`.

- Same `reset()`/`step()` contract as `talon_rl/envs/base_env.py` —
  `training/moppo.py` must not need to change at all to swap envs.
- Robot asset: Unitree A1, sourced from Isaac Lab's shipped
  `isaaclab_assets` — no URDF conversion needed, it's already in the
  release.
- Must satisfy every existing invariant from `CLAUDE.md`:
  - transition dict carries every key `reward._TERM_FUNCS` expects
    (`v_actual`, `v_command`, `obstacle_dist`, `joint_torque`, `joint_vel`,
    `foot_contact_force`, `action`, `prev_action`, `joint_acc`, `obs`)
  - `obs_dim` **excludes** $w$ — the preference vector is appended in
    `training/moppo.py`, never inside the env
- PD conversion ($\tau = K_p(a_t - q) + K_d(\dot a_t - \dot q)$, per
  `docs/mdp.md`) maps onto Isaac Lab's built-in actuator model. Check the
  shipped A1 actuator config's default $K_p$/$K_d$ against chapter3.tex's
  values before assuming they match; override in the env's actuator cfg if
  not.
- `clearance` reward term's signal (obstacle negotiation) is still scripted
  in the dummy env per `docs/mdp.md` — this design does not add a real
  Exteroception-derived signal. `IsaacLabTalonEnv` should provide the same
  kind of placeholder signal `DummyTalonEnv` does, not silently drop the
  term.

## 3. Wiring into `scripts/train_prelim.py`

- Add a `--env {dummy,isaac_lab}` flag (default `dummy`), so the existing
  no-GPU dev workflow and CI keep working unchanged.
- `--env isaac_lab` path imports `IsaacLabTalonEnv` lazily (inside the
  branch, not at module top) so `train_prelim.py --env dummy` still runs
  fine on a machine without Isaac Sim installed.
- "Success" for this milestone = a handful of PPO updates run headless
  against `IsaacLabTalonEnv` without crashing — the same "pipeline runs"
  bar `DummyTalonEnv` already cleared, not a locomotion claim.

## 4. Testing strategy

Isaac Sim can't run inside the normal `pytest tests/` suite — it needs its
own heavyweight `SimulationApp` process and a GPU, neither of which the
existing dummy-env CI path has or should require.

- **In `pytest tests/` (existing 3.12 `.venv`, no GPU needed):** a
  structural test that `IsaacLabTalonEnv` implements every abstract method
  `BaseTalonEnv` requires and reports the right `obs_dim`/`action_dim` —
  importable without actually constructing a live sim. This can only run
  once `isaacsim` is at least importable in that venv, or the test itself
  needs to skip cleanly when it isn't (matches this repo's existing
  no-GPU-available default).
- **Manual smoke run (from `~/isaac-lab-env` venv, on the GPU machine):**
  `python scripts/train_prelim.py --env isaac_lab --updates 5` — this is
  the actual proof the sim wiring works, documented in README as a manual
  step, not part of the automated suite.

## Risks / open questions

- **VRAM ceiling.** RTX 3070 Ti's 8GB is Isaac Sim's minimum-tested tier,
  not the recommended tier (16GB+). Expect to run few parallel envs; this
  caps throughput for any real training later, but doesn't block the
  wiring smoke test this milestone targets. Not solved here — flagged in
  README's existing "known gaps" list if it turns out to matter sooner than
  expected.
- **Actuator $K_p$/$K_d$ mismatch.** Unverified until the A1 actuator config
  is actually read — flagged above, resolved during implementation, not a
  design fork.
- **`isaacsim` import inside the 3.12 `.venv`.** The structural test above
  assumes `isaacsim` is at least importable somewhere pytest can reach. If
  it turns out Python-version-pinning makes that impractical, the
  structural test may need to move to a separate, GPU-venv-only pytest
  invocation instead — small implementation detail, not a design change.
