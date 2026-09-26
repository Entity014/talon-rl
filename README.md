# TALON RL

Preference-conditioned multi-objective locomotion for the **Unitree A1**.

[Teacher architecture](docs/methods/architecture/teacher-architecture.md) · [Train and run](scripts/rl/README.md) · [Experiments](scripts/rl/experiments/README.md) · [Research docs](docs/README.md) · [RL core](scripts/rl/core/README.md)

---

This repo is the research stack for **TALON — Terrain-Adaptive Locomotion via Objective Negotiation**. It contains the Unitree A1 task, reusable RL infrastructure, preference-conditioned model families, simulator-transfer tooling, and the experiment record that explains how the current method was selected.

The active target is a **privileged teacher** that sees robot state, plant/environment factors, and a variable-cardinality objective set. Instead of appending a fixed preference vector to the observation, TALON encodes the requested objectives as a set and uses that representation to realize a preference-specific policy from a learned policy family.

## What TALON is building

The teacher has three inputs:

| input | role |
|---|---|
| robot state + previous action | current control context |
| privileged plant/environment factors | true dynamics/environment context during teacher training |
| objective-weight set | requested multi-objective behavior |

The current architecture is:

```text
robot state + previous action
            |
            v
        State Trunk -----------+
                               |
privileged env factors         |
            |                  |
            v                  |
   Env Factor Encoder ---------+----> Conditional Policy
                               |       params = theta(z_w)
objective-weight set           |               |
            |                  |               v
            v                  |            action
   Set Encoder -> z_w          |               |
            |                  |               v
            v                  +-------> robot dynamics
   Family Hypernetwork
            |
            v
 theta(z_w) = theta_0 + sum_k c_k B_k
```

The critic uses the same state, privileged context, and objective-set semantics and predicts values through a shared objective-query mechanism.

The full architecture, dimensions, and design rationale live in **[docs/methods/architecture/teacher-architecture.md](docs/methods/architecture/teacher-architecture.md)**.

## Where to find things

The links below go to the **README / landing page for each subsystem** first. Those pages explain the local structure and point to the concrete implementation files.

### You want to train or run a policy

| start here | what you will find |
|---|---|
| [RL runner](scripts/rl/README.md) | Training, playback, and sim-to-sim entry points. Start here for `train.py`, `play.py`, and `sim2sim.py`. |
| [RL core](scripts/rl/core/README.md) | Reusable algorithms, rollout/GAE, objectives, preferences, normalization, checkpoints, runtime, and integration boundaries. |
| [Unitree A1 task](talon_rl/tasks/locomotion/a1_env/README.md) | The Isaac Lab locomotion task: scene/MDP configuration, observations, actions, terrain, events, and terminations. |
| [Teacher architecture](docs/methods/architecture/teacher-architecture.md) | The active Phase 1 teacher design: state trunk, privileged factor encoder, objective-set encoder, policy-family hypernetwork, and objective-query critic. |

### You are changing the learning architecture

| start here | what you will find |
|---|---|
| [Models](talon_rl/models/README.md) | Actor/critic architecture families, grouped by mechanism rather than historical experiment ID. |
| [Rewards](talon_rl/rewards/README.md) | Locomotion reward terms, objective grouping, and reward-vector semantics. |
| [Curricula](talon_rl/curricula/README.md) | Command exposure and curriculum state-machine logic. |
| [Optimization](talon_rl/optimization/README.md) | Reusable scalarization, scalar-critic, and optimization helpers. |
| [Wrappers](talon_rl/wrappers/README.md) | Environment/model adapters such as scalar-reward and plant-ensemble wrappers. |
| [TALON package](talon_rl/README.md) | Package-level map showing how models, rewards, tasks, deployment, optimization, and wrappers fit together. |

### You are following the research

| start here | what you will find |
|---|---|
| [Experiments](scripts/rl/experiments/README.md) | Executable research workflows grouped by architecture, baseline, diagnostic, evaluation, and transfer responsibility. |
| [Research docs](docs/README.md) | Master index for the complete research record and the contract → evidence → verdict → closure → thesis chain. |
| [Contracts](docs/contracts/README.md) | Questions, treatment/control definitions, invariants, and pass/fail gates fixed before evaluation. |
| [Verdicts](docs/verdicts/README.md) | Retained conclusions from completed experiments, including failed/blocked branches and causal interpretations. |
| [Closures](docs/closures/README.md) | Branch-level decisions that lock method selection or close a phase so resolved alternatives are not reopened. |
| [Thesis](docs/thesis/README.md) | Method/results/discussion/conclusion synthesis plus traceability and figure/table production notes. |

## Under the hood

The repository is split into three layers.

### `talon_rl/` — reusable task and model package

This is the installed Python package.

```text
talon_rl/
├── assets/          robot assets and configuration
├── curricula/       curriculum and command scheduling
├── deployment/      deployment/runtime and simulator adapters
├── isaaclab/        local Isaac Lab extensions
├── models/          actor/critic architecture families
├── optimization/    reusable optimization helpers
├── rewards/         reward/objective definitions
├── tasks/           robot task definitions
└── wrappers/        environment/model wrappers
```

### `scripts/rl/core/` — reusable RL infrastructure

Training machinery lives here rather than inside the task package.

```text
core/
├── algorithms/
├── checkpoint/
├── diagnostics/
├── envs/
├── experiment_io/
├── integration/
├── modules/
├── normalization/
├── objectives/
├── policies/
├── preferences/
├── rollout/
└── runtime/
```

Environment implementations satisfy the structural `TalonEnv` contract; task packages do not need to inherit from a training-framework base class.

### `scripts/rl/experiments/` — research workflows

Experiments are organized by **what they investigate**, not only by historical phase number.

```text
experiments/
├── architectures/
├── baselines/
├── common/
├── diagnostics/
├── evaluation/
└── transfer/
```

Historical IDs such as `B0`, `V2B`, `C25`, `AI-C2`, and `Phase5-E2` are retained inside stage names, run directories, contracts, and verdicts so the thesis lineage is still auditable.

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

## How the research record works

TALON keeps the experimental history in the repo instead of collapsing it into one final implementation.

```text
contract
   |
   v
experiment implementation
   |
   v
run / artifact evidence
   |
   v
verdict
   |
   v
closure
   |
   v
thesis synthesis
```

That distinction matters:

- **contracts** say what must be tested;
- **experiments** produce the evidence;
- **verdicts** record what the evidence supports;
- **closures** decide what branch remains active;
- **thesis docs** synthesize the retained chain.

Start at **[docs/README.md](docs/README.md)** if you are trying to reconstruct why a design decision exists.

## Current status

Completed research in this repository includes:

- deterministic scalar locomotion and reference-policy reproduction,
- reward-preserving vectorization,
- preference-conditioned PPO / MORL baselines,
- preference-authority and policy-family architecture studies,
- critic representation and semantic-credit diagnostics,
- simulator-transfer and plant-alignment studies,
- controller-transfer and ensemble-robustness studies.

The active next implementation is the **Phase 1 privileged teacher architecture** described above.

Its goal is to establish a controllable, environment-aware policy family before a later student/adaptation stage removes direct access to privileged environment factors.

## TienKung sibling task

[`talon_rl/tasks/manipulation/tienkung_env/`](talon_rl/tasks/manipulation/tienkung_env/) and [`talon_rl/assets/tienkung2_lite/`](talon_rl/assets/tienkung2_lite/) are a sibling research application that reuses TALON's generic infrastructure.

They are **not** part of the defended Unitree A1 locomotion contribution.

## Repository rule of thumb

```text
source folder     = responsibility / mechanism
source file       = concrete responsibility
experiment stage  = historical provenance

docs folder       = document role + research domain
docs filename     = provenance-bearing research identity
```

If you are adding reusable code, it should usually go into `talon_rl/` or `scripts/rl/core/`.

If you are testing a scientific question, it belongs under `scripts/rl/experiments/` with its contract/verdict trail under `docs/`.
