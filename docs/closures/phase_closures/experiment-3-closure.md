# Experiment 3 closure and final validation handoff

Decision locked on 2026-09-20: **Experiment 3 is closed. P99 coefficient
clipping is a rejected intervention. No Experiment 4 or clipping-threshold
search (P97.5, P95, adaptive clipping) is planned.**

## Interpretation of the existing results

Tests 13A/13B remain diagnostic evidence of seed-dependent per-sample
contribution / gradient instability, particularly in efficiency, impact,
and balance. They do not establish the cause of training failure.

The recorded Experiment 3B comparison failed the joint criterion of
mechanism improvement AND no locomotion regression. Layer A did not show
consistent improvement; Layer B reported worse forward locomotion across
all three seeds, including falls, vertical motion, and contact metrics.
The 3A.1 single-checkpoint counterfactual did not adequately predict the
mechanism signature after long-horizon retraining in 3B. This conclusion
is specific to the tested intervention and protocol.

These are the existing experiment verdicts, carried forward from the
research record and the user's decision; this document is not a new
independent reproduction. The original diagnostic preference was not
recorded precisely enough to equate absolute mechanism values across
sessions. Preserve that limitation in the thesis.

Metric wording must distinguish signed mean `v_z`, mean absolute `v_z`,
and RMS `v_z`. A more negative signed mean is not a measurement of mean
absolute vertical speed. Contact count also needs gait context; greater
contact alone does not establish better locomotion.

## Candidate boundary

Evaluate the existing Case 3 baseline family first, preserving all three
training seeds rather than choosing a seed after viewing final results:

`runs/phase1_hipact_dt01_seed{0,1,2}_2026-09-20/checkpoint.pt`

These are candidate checkpoints, not validated final policies. The
additional `seed0_exp3b_baseline` run is a cross-check, not an extra
independent training seed.

The rejected family
`runs/phase1_hipact_dt01_seed{0,1,2}_exp3b_p99_2026-09-20`
must not be used for deployment or sim-to-real.

## Final Simulation Evaluation: protocol defined

The [Final Locomotion Evaluation Protocol v1](final-locomotion-evaluation-protocol.md)
now specifies the matrix, project-defined numerical criteria, preference
contrasts, and objective mapping. It evaluates the current locomotion milestone,
not the full thesis claim. Its detailed definitions supersede the earlier
handoff checklist below. Complete and freeze the execution manifest before
running; no final evaluation has been run as part of this documentation work.

Before collecting final evaluation results, record:

- Checkpoint hashes, code revision and working-tree changes, full effective
  environment/reward configuration, simulator version, timestep, action
  scaling, observation normalization, and reset/termination behavior.
- Required locomotion command range and terrain. Previously tested forward
  commands 0.5 and 0.75 m/s are useful anchors, not an established task
  specification; reverse/standing results cannot establish forward success.
- Exact preference vectors in checkpoint objective order, including floor
  handling. The inspected baseline config uses four objectives:
  progress, efficiency, impact, balance. Do not assume the README's older
  five-objective setup matches these checkpoints.
- Deterministic or stochastic inference, independent evaluation seeds,
  number of lanes, episode duration in seconds, repetitions, warm-up
  handling, and command schedule. Use the same protocol across candidates,
  with no policy updates or training-state adaptation during evaluation.
- Numerical acceptance thresholds tied to the intended locomotion
  specification, and an explicit rule for aggregating across seeds and
  commands. Do not derive these thresholds from the final results.

Report command-component tracking errors, achieved speed/displacement,
falls per episode and per simulated time, episode survival, body height,
tilt, vertical-motion statistics, undesired contacts, foot contact/slip,
and action/torque saturation. Keep per-seed results visible. Account for
early termination and resets so survival-conditioned metrics cannot hide
failures. Retain representative videos and synchronized physical signals.
Rewards alone are not evidence of sufficient locomotion.

At the time of the original handoff, acceptance was pending execution and
scoring under the defined protocol. The completed result is recorded below.

### Execution result (2026-09-20)

The 14,400-episode evaluation is complete. The current baseline family fails
the locomotion milestone and all four preference contrasts are
NOT-DEMONSTRATED. See the [final evaluation report](../artifacts/final_locomotion_eval_v1/final-report.md)
and machine-readable score. Do not freeze or deploy this candidate. This result
opens a narrowly evidence-backed objective/reward revision cycle; it does not
reopen Experiment 3 or change the rejected P99 verdict.
The scope is frozen in the [Revision R1 design](reward-objectives-revision-r1-design.md).

## Objective / sub-reward audit and decision

After final evaluation, map each implemented reward/sub-reward to its
intended behavior and an independent physical metric. Check progress
versus evaluation tracking, reward exploits, and the conceptual placement
of hip activation under balance. Inspect actual checkpoint configuration
and implementation rather than assuming documentation is current.

An audit finding does not automatically require a reward edit:

- If behavior meets the locked specification, freeze the candidate and
  reward definition; document remaining conceptual limitations.
- If behavior fails and evidence supports a reward loophole or an
  objective/behavior mismatch, revise only the supported reward terms,
  retrain, and re-evaluate. Failure by itself does not identify a reward
  defect or authorize indiscriminate reward tuning.

Consolidate the final report/thesis results before considering deployment
or sim-to-real. Keep the rejected P99 family excluded throughout.
