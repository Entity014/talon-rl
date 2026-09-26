# TALON RL

**Terrain-Adaptive Locomotion via Objective Negotiation**

Preference-conditioned multi-objective locomotion for the **Unitree A1**.

[Architecture](docs/methods/architecture/teacher-architecture.md) · [Train and run](scripts/rl/README.md) · [Experiments](scripts/rl/experiments/README.md) · [Research](docs/README.md) · [RL core](scripts/rl/core/README.md) · [Package](talon_rl/README.md)

---

**TALON is a locomotion research stack for teaching one quadruped policy to negotiate multiple objectives, adapt to changing robot dynamics, and preserve controllable behavior across simulation and deployment conditions.**

Instead of training one controller for one fixed trade-off, TALON conditions a policy family on an objective-weight set. The current work extends that controller with privileged plant context for a teacher–student adaptation pipeline.

## What TALON does

| capability | what it means |
|---|---|
| **Objective trade-offs** | Change the objective weights and expose a different locomotion behavior without training a separate controller for every trade-off. |
| **Plant-aware control** | The privileged teacher conditions on plant/environment context such as mass, CoM, friction, terrain, and actuator strength. |
| **Policy families** | A preference-set latent realizes a policy from a learned low-dimensional family instead of only appending a preference vector to the observation. |
| **Transfer analysis** | The research stack includes deployment-equivalence, Isaac → MuJoCo, plant-alignment, controller-robustness, and ensemble-robustness studies. |

> Future result figures, rollout GIFs, and comparison videos belong here — close to the capabilities they demonstrate.

## Why TALON?

Most locomotion policies are trained around one fixed objective balance. TALON studies a harder question:

> Can one controller expose a usable trade-off interface while remaining stable under learning, changing dynamics, and simulator transfer?

That question drives both the architecture and the experiment record in this repository.

## Architecture at a glance

```text
robot state + previous action
            |
            v
        State Trunk -----------+
                               |
privileged plant factors       |
            |                  |
            v                  |
    Env Factor Encoder --------+----> Conditional Policy
                               |       params = theta(z_w)
objective-weight set           |               |
            |                  |               v
            v                  |            action
       Set Encoder             |
            |                  |
           z_w                 |
            |                  |
            v                  |
   Family Hypernetwork --------+
```

The critic uses the same state, privileged context, and objective-set semantics through a shared objective-query value mechanism.

→ **[Full teacher architecture](docs/methods/architecture/teacher-architecture.md)**

## Where to find things

### You want to train or run TALON

| start here | what you will find |
|---|---|
| [RL runner](scripts/rl/README.md) | Training, playback, evaluation, and sim-to-sim entry points. |
| [Unitree A1 task](talon_rl/tasks/locomotion/a1_env/README.md) | Isaac Lab locomotion environment, observations, actions, terrain, events, and terminations. |
| [Teacher architecture](docs/methods/architecture/teacher-architecture.md) | Current model design and teacher-stage data flow. |

### You are changing the method

| start here | what you will find |
|---|---|
| [Models](talon_rl/models/README.md) | Actor/critic and preference-conditioning architectures. |
| [Rewards](talon_rl/rewards/README.md) | Objectives and reward semantics. |
| [Optimization](talon_rl/optimization/README.md) | Scalarization and critic/optimization helpers. |
| [RL core](scripts/rl/core/README.md) | Shared algorithms, rollout, objectives, preferences, normalization, checkpointing, and runtime. |

### You are following the research

| start here | what you will find |
|---|---|
| [Research docs](docs/README.md) | The map of methods, contracts, verdicts, closures, protocols, and thesis synthesis. |
| [Experiments](scripts/rl/experiments/README.md) | Executable research workflows grouped by responsibility. |
| [Contracts](docs/contracts/README.md) | What was fixed before an experiment was evaluated. |
| [Verdicts](docs/verdicts/README.md) | What the evidence supports. |
| [Closures](docs/closures/README.md) | Which branches are closed and which direction remains active. |
| [Thesis](docs/thesis/README.md) | Thesis-facing method, results, discussion, conclusion, and traceability. |

## Under the hood

```text
talon_rl/
├── models/
├── rewards/
├── tasks/
├── deployment/
├── optimization/
└── wrappers/

scripts/rl/
├── core/
└── experiments/

docs/
├── methods/
├── contracts/
├── verdicts/
├── closures/
└── thesis/
```

Reusable mechanisms live in `talon_rl/` and `scripts/rl/core/`. Scientific questions live in `scripts/rl/experiments/`, with their contracts, verdicts, and closures under `docs/`.

## Research status

**Current**

- **Phase 1 — privileged teacher architecture**

**Completed foundations**

- ✓ preference-conditioned policy-family research
- ✓ preference-authority and semantic diagnostics
- ✓ deployment-equivalence analysis
- ✓ Isaac → MuJoCo transfer studies
- ✓ plant-alignment and ensemble-robustness studies

**Next**

`teacher implementation → adaptation student → deployment validation`

For the experiment lineage and retained evidence, start at **[docs/README.md](docs/README.md)**.

## Quickstart

Python 3.10+ is required.

```bash
git clone https://github.com/Entity014/talon-rl.git
cd talon-rl

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

Main entry points:

```bash
python scripts/rl/train.py
python scripts/rl/play.py
python scripts/rl/sim2sim.py
```

Isaac Lab training requires the project's Isaac Lab / Isaac Sim environment and a compatible GPU setup.

## Thesis and research record

The thesis-facing documents live under [`docs/thesis/`](docs/thesis/README.md). The complete evidence chain is indexed from [`docs/README.md`](docs/README.md).

A formal citation file and repository license have not yet been added; when they are defined, they should live at the repository root rather than being duplicated across subsystem READMEs.
