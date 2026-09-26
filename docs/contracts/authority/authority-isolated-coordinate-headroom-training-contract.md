# Authority-Isolated Coordinate-Aware Headroom Training Contract

Status: **FROZEN BEFORE TRAINING**
Date: 2026-09-25

## Frozen components

- authority-isolated actor architecture: unchanged
- preference-only generated family path: unchanged
- wide critic architecture: unchanged
- GAE lambda: 0.95
- preference schedule: unchanged
- Foundation/reset-diverse support: unchanged
- PPO objective and stochastic action/log-prob contract: unchanged

## Sole treatment

Add a per-action raw-logit tail penalty to the actor loss:

    L_total = L_PPO + lambda_tail * L_tail

    L_tail = mean_j [ relu(|z_j| - tau_j)^2 / (tau_j^2 + eps) ]

where z is the deterministic actor pre-tanh mean on the PPO batch.

The per-coordinate tau_j values are frozen from the robust u50 actor on the fixed probe corpus, using the 95th percentile of |z_j|. No residual-lane or FL-hip data are used to set tau_j.

Control is the exact same training path with lambda_tail = 0.

## Frozen tau source

    runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json

## Short paired screen

Start from the same u75 authority-isolated actor and the same repaired wide critic.

Arms:
- CONTROL: lambda_tail = 0
- TREATMENT: lambda_tail = 0.01

Initial screen:
- 10 actor updates
- identical random seed
- identical reset/support/preference schedule
- critic updated identically in both arms
- save update 0 and update 10

The treatment coefficient is fixed before execution and is not tuned on residual-lane outcome.

## Gates

Authority preservation on fixed probe:
- pairwise preference separation retention >= 0.90 versus each arm's update-0 value
- simplex-tangent action Jacobian retention >= 0.90 versus each arm's update-0 value

Tail mechanism:
- normalized tail loss decreases in treatment relative to treatment update 0
- fraction of coordinates exceeding tau decreases relative to treatment update 0
- no requirement for zero saturation

Support-margin protection:
- suite3 / C / lane0 must survive
- FL-hip action during t=4..7 must not fall below the empirically causal support range established by the residual audit; report mean/min, do not train on this lane directly

Global robustness:
- semantic suites 2-3 minimum survival >= 0.95
- failed lanes lower than control

Decision:
- survival improves + authority preserved -> authorize longer confirm
- survival improves but authority <0.90 -> fail
- tail decreases without robustness improvement -> mechanism insufficient
- robustness worsens or FL-hip support margin is lost -> fail

AI-C2/H2 remain blocked until a treatment passes the robustness and authority gates.
