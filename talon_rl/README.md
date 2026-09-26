# talon_rl

<!-- nav:start -->
[TALON RL](../README.md)
<!-- nav:end -->


Installed package for Talon robot assets, task definitions, reusable models, rewards, curricula, deployment helpers, and experiment-facing wrappers.

## Layout

```text
assets/       robot asset/config definitions
curricula/    pure curriculum and command-schedule logic
deployment/   deployment runtimes and simulator adapters
envs/         installed environment wrappers
isaaclab/     local Isaac Lab extensions
models/       actor/critic model families
pivots/       reusable pivot-experiment primitives
rewards/      reward functions and objective-vector transformations
tasks/        real task/environment definitions
wrappers/     reusable environment/model wrappers
config.py     package-level configuration dataclasses
```

## Design Rules

1. Keep top-level Python files minimal: package entry points and broad configuration only.
2. Group implementation modules by responsibility, not experiment chronology.
3. Preserve class/function names when they carry thesis traceability.
4. Small variants from the same architecture/reward family may share a module.
5. Isaac-specific task wiring belongs under `tasks/`; pure math/state-machine logic should stay outside task modules when practical.
6. Structural refactors must preserve numerical behavior and checkpoint/state-dict compatibility.

## Adding New Code

Place new model architectures under `models/`, reward/objective semantics under `rewards/`, curriculum logic under `curricula/`, and deployment/runtime adapters under `deployment/`. Avoid adding new implementation files directly at the package root.
