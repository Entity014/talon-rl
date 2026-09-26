# Consolidated locomotion failure analysis

Status: **decision document — no new intervention authorized**  
Scope: frozen v1 baseline, R1 reward revision, G1 command-exposure revision,
and D1 deterministic-mean diagnostics. This document does not revise any
frozen evaluation result.

## Milestone status

The frozen locomotion milestone remains **FAIL / INADEQUATE**. Neither R1 nor
G1 produced a successful deterministic 22-second final-evaluation episode at
uniform preference across all three seeds. Preference contrasts consequently
remain **NOT-DEMONSTRATED**, because the locomotion gate never opened.

| branch | intended change | outcome |
|---|---|---|
| v1 | frozen baseline | FAIL / INADEQUATE |
| P99 | gradient-clipping intervention | did not provide a usable fix |
| R1 | reward revision, including gated alive bonus | FAIL / INADEQUATE |
| G1 | evaluator-aligned command exposure | FAIL / INADEQUATE |
| D1 | deterministic-mean mechanism diagnosis | no trainable intervention identified |

Primary frozen records are the [v1 report](../artifacts/final_locomotion_eval_v1/final-report.md),
[R1 report](../artifacts/final_locomotion_eval_r1/final-report.md), and
[G1 report](../artifacts/final_locomotion_eval_g1/final-report.md).

## What the diagnostics established

The R1 mismatch audit showed a real, seed-dependent execution and
generalization gap. Seed1 could survive under some stochastic executions while
its deterministic mean failed almost immediately; seed0 was strongly sensitive
to command condition; seed2 was sensitive to command and observation history.
The failures of seed0 and seed1 occurred during the zero-command settling
period, when the R1 alive gate was open. This rules out the gate being off at
evaluation time as the direct trigger, although it does not prove reward
causality false.

The audit also corrected an earlier interpretation of training telemetry:
reported mean episode length was current lane age after terminated lanes reset,
not completed episode length. It cannot be compared directly with final-eval
first-fall time.

Command exposure was the strongest demonstrated distribution contrast:
training-style random commands improved short-horizon survival relative to the
forced evaluator schedule. Reset differences were not dominant in tested
contrasts. Terrain-only and physics-randomization-only conditions did not
rescue deterministic fixed-command behavior. These are diagnostic results, not
post-hoc changes to the evaluator.

## What the interventions ruled out

R1 did not establish that reward bootstrap starvation is the principal cause.
Its reward change failed, but the deterministic collapse and later diagnostics
showed several execution and distribution factors that R1 did not isolate.
Coefficient retuning is therefore not supported by R1 alone.

G1 tested one clean exposure-only change: 25% per-episode exact zero-command
holds for 200 steps followed by a training-range command, while reward, PPO,
architecture, simulator timing, physics distributions, and final evaluation
remained frozen. It failed in all seeds under final evaluation, including loss
of seed2 survival present in R1. Thus command exposure is a demonstrated
diagnostic factor but the tested exposure-only intervention is not sufficient.
No probability, hold-duration, or command-schedule sweep is warranted from
that failure.

D1 examined why stochastic execution could sometimes avoid early failure while
the deterministic actor mean could not. Small same-state probes suggested that
magnitude and direction may affect early tilt, but those patterns did not
replicate as a simple global rule at 100 steps. Across 16 instrumented RNG
streams and 512 stochastic trajectories, survivor-versus-fall action deltas
did not yield a fixed joint or subspace signature consistent across streams.
The strongest joint-level sign consistency was 75%, below the required
repeatability threshold. Global shrinking, clipping, and targeted joint
regularization would therefore be post-hoc choices rather than evidence-backed
single-variable interventions.

## Current diagnosis

The best supported diagnosis is:

> Deterministic-mean fragility is a distributed, state-dependent action-geometry
> problem under the frozen evaluation conditions. It is associated with command
> exposure and command-history sensitivity, but no tested reward, exposure, or
> fixed action-subspace intervention has been sufficient to repair it.

This diagnosis does not claim that reward, terrain, physics randomization, or
stochasticity are irrelevant to learning. It only states what the completed
tests do and do not support as the next controlled intervention.

## Closed branches and guardrails

- **P99:** closed; no usable fix from the tested gradient-clipping path.
- **R1:** closed FAIL; reward causality remains unproven; R2 remains closed.
- **G1:** closed FAIL; do not sweep zero-hold probability or duration.
- **D1:** closed diagnostic; do not retrain global shrink/clipping or a fixed
  joint/subspace regularizer.
- The frozen final evaluator remains the milestone evaluator. Diagnostic probes
  supplement it and must not replace its deterministic execution or gates.

## Decision point

No further local intervention is authorized by current evidence. The next work
is a thesis-level decision, not another coefficient or diagnostic sweep:

1. Treat the reproducible failure analysis and negative intervention results as
   a research contribution, then move to another proposal question; or
2. Define a new baseline architecture or training formulation as a new project
   phase, with a fresh causal rationale, frozen design, and clean baseline
   protocol before training.

Any future work on preference conditioning, terrain, or payload requires a
locomotion baseline that clears the relevant capability gate first.

## Supporting artifacts

- [R1 mismatch audit](../artifacts/r1_mismatch_audit/audit-report.md)
- [D1 design analysis](../artifacts/r1_mismatch_audit/d1_design_analysis.md)
- [D1.2b aggregate](../artifacts/r1_mismatch_audit/d1_2b_aggregate.json)
- [G1 freeze record](../artifacts/g1_freeze/G1_FREEZE.md)
