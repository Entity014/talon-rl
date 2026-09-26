# Phase 5 — Plant-Ensemble Semantic Robustness Contract

Status: **PREDECLARED — ENSEMBLE FREEZE BEFORE TRAINING**
Date: 2026-09-26

## 1. Scientific question

Can the frozen Phase-1 MORL controller formulation be trained over a preregistered distribution of plausible plant realizations such that preference-conditioned semantics remain valid across plant variation, without sacrificing source-domain authority, critic validity, or locomotion viability?

Phase 5 is a new method branch. It does not rewrite D3, T1-T4, or Phase 4 results.

Frozen prior evidence:
- D3 zero-adaptation transfer: PARTIAL TRANSFER.
- T1: semantic transfer failure arises after rapid dynamics-driven state-visitation divergence.
- T2-T4: no single actuator/passive correction restored source-like closed-loop semantics.
- Phase 4: distributed multi-step feedback amplification is strongly supported.
- No compact Jacobian, Lipschitz, or scalar sensitivity predictor was validated.

## 2. Core hypothesis

The remaining transfer limitation is distributed and multi-step.

Therefore the authorized intervention class is not a local sensitivity penalty. It is direct training over a bounded plant ensemble:

    p ~ P_plant
    a_t = pi(s_t, w)

with the objective that semantic behavior remains valid across sampled plant realizations.

Success requires semantic robustness, not merely survival robustness.
## 3. Non-goals

Phase 5 does NOT authorize:
- arbitrary domain randomization;
- unconstrained parameter sweeps;
- MuJoCo-specific policy fine-tuning;
- objective-specific rescue losses;
- Jacobian/Lipschitz penalties;
- reward redefinition;
- changing the Phase-1 objective vocabulary;
- changing the canonical deployment observation/action contract;
- using D3 target semantic outcomes for checkpoint selection.

The D3 zero-adaptation result remains the frozen baseline.

## 4. Frozen controller-side contract

Unless explicitly opened by a later Phase-5 sub-contract, retain:
- authority-isolated actor architecture;
- Phase-1 objective order T/A/O/S;
- canonical 48-D observation contract;
- canonical action mapping q_target = q_default + 0.25 a;
- 50 Hz policy rate;
- simplex-edge authority-retention mechanism;
- validated PPO semantics;
- validated critic substrate;
- no objective-specific actor or critic branch.

Phase 5 changes only the training plant distribution in the first authorized treatment.

## 5. Plant-ensemble construction principle

The ensemble must be evidence-derived from residual source-target mismatch, not chosen post hoc to rescue MuJoCo.

Allowed parameter families are limited to quantities already implicated or bounded by D3/T1-T4/Phase4 audits:
- total/link mass and inertia;
- inertial-frame / COM offsets;
- passive joint damping, friction, and armature;
- actuator response / torque-speed realization;
- contact/compliance parameters;
- integration/contact-solver surrogate parameters representable in the source simulator.

Every family requires a frozen nominal value and bounded perturbation range before training.
## 6. Phase 5 gates

Phase 5 is split into five gates:

    P5-E0  ensemble identifiability / freeze
    P5-E1  source-domain no-regression baseline
    P5-E2  plant-ensemble training screen
    P5-E3  held-out plant semantic robustness
    P5-E4  cross-engine MuJoCo validation

No gate may be skipped.

## 7. P5-E0 — Ensemble identifiability and freeze

Purpose: define P_plant without training feedback.

For each candidate plant parameter family:
1. establish the source nominal value;
2. establish the target/reference discrepancy when measurable;
3. define a bounded training range using source-target evidence or a predeclared engineering tolerance;
4. define sampling distribution;
5. define whether parameters are sampled independently or in coupled blocks.

Forbidden:
- choosing a range because it makes MuJoCo semantics improve;
- expanding ranges after seeing training results;
- adding a parameter family after E2 results without a new contract.

E0 output must include a machine-readable ensemble manifest and hashes of all source model files used to derive it.

Training is blocked until E0 is frozen.
## 8. E0 ensemble sanity probes

Before actor training, sample the ensemble and verify:
- no invalid/non-finite physics;
- joint/actuator limits remain physically meaningful;
- reset feasibility remains bounded;
- fixed canonical stance remains viable for a preregistered fraction of samples;
- perturbations span but do not grossly exceed the measured source-target residual envelope.

Report marginal and joint distributions.

If the ensemble itself creates widespread nonphysical failure, E0 fails and must be redesigned before any policy training.

## 9. P5-E1 — Source-domain no-regression baseline

Run the frozen canonical Phase-1 controller on:
- nominal source plant;
- sampled in-ensemble plants;
- held-out perturbation plants reserved before training.

Measure:
- survival;
- T/A/O/S endpoint semantics;
- center compromise;
- continuum metrics;
- pairwise/tangent authority;
- critic validity.

E1 is characterization only. No training or tuning.

Its purpose is to establish how much of the ensemble is already inside the controller's validity envelope.
## 10. P5-E2 — Plant-ensemble training screen

Treatment:
- initialize from the frozen Phase-1 training source defined by a dedicated E2 initialization manifest;
- sample plant p ~ P_plant during training;
- keep objective/preference training protocol unchanged;
- keep semantic validation out of checkpoint selection.

Control:
- exact same continuation budget on the nominal source plant only.

Primary question:

> Does ensemble exposure improve semantic robustness across plant variation without destroying the known source-domain controller properties?

The first E2 run is a bounded screen, not a sweep.

Only one preregistered ensemble distribution and one training budget are allowed in the first screen.
## 11. E2 primary gates

Treatment must satisfy all of the following relative to the frozen source reference and matched nominal control:

### Source-domain preservation
- global pairwise authority: PASS;
- tangent authority: PASS;
- critic validity: PASS;
- T semantics: PASS;
- O semantics: PASS;
- A reported under the D1 seed-sensitive interpretation;
- S reported without being a treatment success requirement;
- nominal-source survival within the frozen robustness gate.

### In-ensemble robustness
Across sampled plant realizations:
- finite inference and valid actions;
- survival above the preregistered ensemble floor;
- T/O semantic direction preserved on the required fraction;
- A/S characterized identically, not patched.

### No hidden specialization
A single actor/critic checkpoint must serve all plant samples.
Plant parameters are not provided to the policy in the first Phase-5 treatment.
## 12. P5-E3 — Held-out plant generalization

E3 uses plant realizations never sampled during E2 training.

Held-out plants must be frozen before E2 starts.

Evaluate:
- survival;
- T/A/O/S endpoint semantics;
- center compromise;
- continuum monotonicity;
- endpoint-between;
- preference authority;
- critic validity;
- trajectory divergence from the nominal source.

Primary claim boundary if E3 passes:

> Training over a bounded, preregistered plant ensemble improves semantic robustness to unseen plant variations while preserving the validated source-domain MORL properties.

Do not claim arbitrary dynamics generalization.
## 13. P5-E4 — Cross-engine MuJoCo validation

E4 reuses the frozen D2 deployment boundary and D3 MuJoCo protocol.

No MuJoCo fine-tuning is allowed before the first E4 result.

Compare:
- D3 zero-adaptation canonical controller;
- Phase-5 ensemble-trained controller.

Primary outcomes:
- survival;
- T semantic recovery;
- A/O/S semantic transfer;
- center compromise;
- continuum monotonicity;
- endpoint-between;
- action saturation;
- height and base-dynamics envelope.

E4 success requires semantic improvement without source-domain no-regression failure.

A MuJoCo-only improvement is insufficient if source-domain semantics regress.
## 14. Interpretation matrix

Possible outcomes:

1. E2 source preserved, E3 improves, E4 improves
   -> strong support for plant-ensemble semantic robustness.

2. E2 source preserved, E3 improves, E4 unchanged
   -> robustness improved within the modeled ensemble, but cross-engine residual remains out-of-distribution.

3. E2/E3 improve survival but not semantics
   -> locomotion robustness is not semantic robustness.

4. Source semantics regress during E2
   -> ensemble training formulation fails no-regression.

5. E3 fails despite E2 success
   -> ensemble overfit / insufficient generalization.

6. E4 improves only after MuJoCo-specific adaptation
   -> cannot be credited to zero-adaptation plant-ensemble robustness.

## 15. Stop rules

Stop Phase 5 method escalation if:
- E0 cannot define a defensible bounded ensemble;
- E2 destroys source-domain T/O semantics or global authority;
- E2 only improves survival while semantic transfer remains unchanged;
- E3 shows no held-out benefit;
- a proposed follow-up requires MuJoCo-specific parameter tuning or objective-specific repair without a new contract.

Do not respond to a failed gate by expanding the ensemble or adding parameter families post hoc.
## 16. Relation to prior findings

Phase 5 is motivated by the following evidence chain:

    pointwise parity
        != closed-loop equivalence

    local plant parity
        != semantic transfer

    local actor sensitivity
        != semantic-transfer predictor

    distributed residual plant mismatch
      + repeated policy feedback
        -> finite-horizon trajectory amplification
        -> semantic separation

The Phase-5 intervention therefore operates at the distribution-of-plants level rather than through a scalar local robustness penalty.

## 17. Claim boundary

Phase 5 may establish robustness over a bounded plant family.

It may not establish:
- universal sim-to-real robustness;
- arbitrary plant invariance;
- unseen-objective generalization;
- 4/4 semantic validity across all seeds;
- hardware safety.

D4 remains blocked until the applicable Phase-5 transfer gate passes.

## 18. Immediate next action

NEXT = P5-E0 only.

Do not train yet.

Create the plant-ensemble manifest by auditing the source Isaac model, frozen MuJoCo T4 model, and the residual discrepancy evidence already produced in T1-T4/Phase4.

Only after E0 ranges, coupling rules, held-out plants, and provenance are frozen may P5-E1/E2 begin.
