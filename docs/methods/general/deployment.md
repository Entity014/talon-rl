# TALON Deployment Boundary

> **Historical / superseded for the final Phase-1 controller.** The canonical authority-isolated Phase-1 deployment boundary is now defined by `docs/contracts/transfer/phase1-d2-deployment-equivalence-contract.md` and `docs/verdicts/transfer/phase1-d2-deployment-equivalence-verdict.md`. In particular, the validated final controller uses the actor-only eager-state CUDA runtime; the older TorchScript `play.py` export path below is not authorized for that canonical controller because D2 found closed-loop sensitivity to sub-micro compiled-graph numerical differences.

Experiment 3's P99 coefficient-clipping checkpoints are rejected and must
not be deployed or used for sim-to-real. Baseline candidates still require
final simulation validation; see the [locked decision and validation
handoff](experiment-3-closure.md).
The [final locomotion protocol](final-locomotion-evaluation-protocol.md)
defines the current simulation milestone; passing it does not establish
hardware readiness or completion of thesis-scale validation.

The repository currently exports a policy and provides a guarded inference
boundary. It does not contain a Unitree motor transport, real-time scheduler,
or emergency-stop implementation.

## Train for sim2sim

The current Isaac Lab adaptation encoder is privileged and has no trained
student estimator for MuJoCo or hardware. Train a deployable actor with the
encoder disabled so its observation width matches the current MuJoCo adapter:

```bash
PYTHONPATH=.:scripts python scripts/rl/train_prelim.py \
  --env isaac_lab --no_encoder --torch_compile --action_scale 0.15 --stand_phase_s 2.0 \
  --num_envs 256 --updates 1000 \
  --logs_root runs --run_name a1_deployable --save_every 100
```

Use the same `--no_encoder` flag when loading that checkpoint for export:

```bash
PYTHONPATH=.:scripts python scripts/rl/play.py \
  --env dummy --no_encoder --checkpoint runs/a1_deployable/checkpoint.pt \
  --export artifacts/a1_deployable.pt
PYTHONPATH=.:scripts python scripts/rl/sim2sim.py \
  --policy artifacts/a1_deployable.pt --steps 1000
```

The sim2sim adapter currently supports one policy stack and assumes the Isaac
Lab and MuJoCo A1 joint order matches. That assumption must be verified before
using its result as evidence for hardware deployment.

## Export and guarded inference

Export an actor from a validated checkpoint with `play.py`:

```bash
PYTHONPATH=.:scripts python scripts/rl/play.py \
  --checkpoint runs/<run>/checkpoint.pt \
  --env isaac_lab --export artifacts/talon_policy.pt
```

`rl.core.deployment.PolicyRuntime` should be called by the hardware adapter.
It verifies the observation batch width and finiteness, verifies the exported
action shape, and clips actions to the configured final bounds.

This is only a policy boundary. The hardware adapter must still implement:

- sensor acquisition and observation construction in the trained field order;
- fixed-rate control, timeout handling, and stale-sensor detection;
- joint, velocity, torque, and current limits;
- fall detection, safe-stop behavior, and a physical emergency stop;
- logging and a rollback path for the policy artifact.

Do not run an exported policy on a robot until sim-to-sim validation and
tethered hardware tests have passed. The current MuJoCo path is a mechanism
smoke test, not that validation.
