# Gate 0 — locomotion substrate and thesis-pipeline reconciliation

Status: **DECISION PENDING — no new training authorized**

## Why this gate exists

The defended thesis pipeline identifies the Multi-Objective preference-conditioned policy as the principal contribution. Adaptation/RMA and Exteroception are supporting infrastructure. Its intended sequence is Phase-1 MOPPO, adaptation distillation, then validation/deployment. B0/B1 were a later substrate investigation and must not silently replace that thesis scope.

The B0/B1 evidence is still relevant: deterministic locomotion can emerge transiently; P2 improves policy preservation; S1 improves local stability in selected basins; and R1 improves representation/stability. However, no B1 branch passed the three-seed deterministic gate. This is evidence about the locomotion substrate, not a reason to declare the original MOPPO contribution complete or failed by implication.

## Prospective Gate-0 question

Before returning to MOPPO/RMA, what minimum locomotion substrate is required by the actual thesis deployment claim?

The answer must be written and frozen **before** any new training. Existing B0/B1 checkpoints are evidence only; they cannot satisfy a newly chosen gate retrospectively.

## Non-negotiable Gate-0 contract

The selected gate must specify:

- the deployment policy path and observation contract;
- whether the claim is deterministic actor-mean deployment or stochastic execution;
- nominal versus varied environment conditions and command range;
- independent frozen evaluation states and horizon;
- seed count and aggregation rule;
- survival, tracking, tilt/height, contact, and numerical-failure thresholds;
- whether the gate is a substrate sanity gate or a thesis-level locomotion result;
- checkpoint/update used for the verdict;
- artifact, hash, and lifecycle requirements.

The gate must remain independent of optimizer state, preference state, privileged evaluation inputs, and training-lane stochastic return. A failed gate triggers a formulation decision; it does not authorize threshold relaxation, checkpoint cherry-picking, or retrospective promotion of an old run.

## Two legitimate decisions

### A. Strict deterministic substrate gate

Retain the B0-style requirement: all three seeds must pass the deterministic actor-mean gate before any MOPPO/RMA work. If selected, this authorizes a new architecture-level locomotion baseline phase, with a new formulation and freeze record—not another P1/P2/S1/R1 patch.

### B. Pipeline-aligned substrate gate

Define a smaller gate that matches the actual base-policy role in the thesis pipeline, for example a deployment-relevant nominal capability and explicit hand-off requirements to MOPPO/RMA. This is not a relaxation of old B0 results: it is a new prospective contract, frozen first, followed by a fresh rerun. Passing it authorizes the original sequence:

`Base Policy → command range/transitions → domain variation → RMA adaptation → MOPPO/preference conditioning`.

Privileged teacher factors may be used where the RMA design calls for them, but the deployed student/inference contract must be explicit. Privileged cues must not be silently added to a flat baseline and then treated as the RMA student.

## Current decision boundary

Until A or B is selected and frozen, keep B0.2, MOPPO, RMA training, and additional B1 branch composition closed. The next action is a thesis/pipeline decision, not another RL run.
