# S Phase-Localized H16 Sensitivity Verdict

Status: **FROZEN — SHORT-HORIZON S SIGNAL IS REAL BUT NOT PHASE-LOCALIZED OR TRANSFERABLE; SHORT-HORIZON TARGET MINING STOPPED**
Date: 2026-09-25

## Question

Can the meaningful H16 closed-loop S sensitivity be localized to a pre-outcome source phase that:
- remains a valid small-epsilon derivative;
- predicts final full-horizon S semantic sign;
- transfers across held-out reset suites?

If yes, a compact temporal semantic repair target may be justified.
If no, H16 signal is real but context/phase dependent and should not be turned into a local training target.

## Checkpoint

Validated u30:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

No training updates were performed.

## Source phases

    t = 0, 4, 8, 12, 16

At each source:
- reset and replay the center policy to the source phase;
- preserve the exact simulator trajectory/history;
- branch only after the source state;
- preserve the same previous action entering the action-rate reward;
- use H=16 fixed.

Finite-difference direction:

    d_S = w_S - w_C
        = [-.15, -.15, -.15, +.45]

Small epsilon:

    primary  .005
    controls .01, .02

No H32 information is used for phase selection.

## Source-state equality

Across every +/- epsilon branch and every replay:

    maximum source observation/history mismatch = 0.0

Therefore phase comparisons are not confounded by different source simulator states or previous-action history.

## Pooled phase results

Primary epsilon = .005.

### t0

    AUC correct-vs-wrong          .765
    Spearman with final Delta S   .429
    p                             .014
    sign agreement                .750
    mean LOSO balanced accuracy   .680

Finite-difference consistency:

    eps .01 vs .005:
      Pearson                     .845
      sign agreement              .969
      median relative difference  .053

    eps .02 vs .005:
      Pearson                     .737
      sign agreement              .906
      median relative difference  .056

This is the strongest pooled phase, but not uniformly transferable.

### t4

    AUC                           .583
    Spearman                      .111
    sign agreement                .563
    mean LOSO BA                  .493

Derivative consistency also worsens.

Decision:
    NULL / non-useful.

### t8

    AUC                           .636
    Spearman                      .381
    p                             .032
    sign agreement                .656
    mean LOSO BA                  .524

There is signal, but held-out transfer remains weak.

### t12

    AUC                           .490
    Spearman                      .069
    sign agreement                .500
    mean LOSO BA                  .443

Decision:
    NULL.

### t16

    AUC                           .696
    Spearman                      .413
    p                             .019
    sign agreement                .656
    mean LOSO BA                  .574

Finite-difference consistency:

    eps .01 vs .005:
      Pearson                     .757
      sign agreement              .906
      median relative difference  .187

    eps .02 vs .005:
      Pearson                     .462
      sign agreement              .813
      median relative difference  .243

The phase contains meaningful pooled signal, but its derivative is less stable and transfer remains insufficient.

## Per-suite transfer

### t0

    suite 0:
      AUC   .714
      rho   .429
      sign  .750

    suite 1:
      AUC   .333
      rho  -.405
      sign  .375

    suite 2:
      AUC  1.000
      rho   .786
      sign  .875

    suite 3:
      AUC  1.000
      rho   .881
      sign 1.000

The apparently strong pooled t0 result is driven heavily by suites 2 and 3.

Suite 1 reverses the relation.

### t16

    suite 0:
      AUC  0.000
      rho -.524
      sign .375

    suite 1:
      AUC  .667
      rho  .619
      sign .750

    suite 2:
      AUC  .750
      rho  .262
      sign .750

    suite 3:
      AUC  .750
      rho  .714
      sign .750

Again, the signal is not universal.

Suite 0 reverses strongly.

## Expected phase-localization pattern

The predeclared desired pattern was approximately:

    t0    weak
    t4    stronger
    t8    strong
    t12   strong
    t16   perhaps beginning to bifurcate

Observed:

    t0    strongest pooled
    t4    weak
    t8    moderate
    t12   null
    t16   moderate/strong pooled but unstable and non-transferable

No coherent pre-outcome temporal localization emerges.

## Interpretation

Supported:

    H16 closed-loop S sensitivity contains real information.

    Some source phases correlate with final semantic outcome.

    The effect is stronger than the one-step local semantic-gradient signal.

Not supported:

    one transferable pre-outcome source phase;
    one monotonic temporal build-up of semantic credit;
    a compact phase-local H16 target;
    universal cross-suite H16 semantic derivative.

The S semantic effect is therefore better described as:

    longer-horizon closed-loop sensitivity
    that remains context- and phase-dependent.

This is not equivalent to a stable local semantic credit signal.

## Decision

    H16 signal exists                       YES
    H16 phase-localization                  FAIL
    cross-suite transfer                    FAIL
    compact local repair target             NOT AUTHORIZED
    further short-horizon phase mining      STOP
    H2a                                     remains FAIL
    H2b                                     NOT AUTHORIZED

## Research implication

The tested hierarchy is now:

    reward/proxy mismatch                 rejected
    lack of authority                     rejected
    simple context rule                   rejected
    one-step semantic gradient            rejected
    H8 closed-loop sensitivity            weak but real
    H16 closed-loop sensitivity           meaningful
    phase-local H16 transferable target   rejected
    H32 local derivative                  invalid / basin-sensitive

Therefore the remaining S semantic inconsistency is a multi-step closed-loop phenomenon that is not captured by a compact local target in the tested representations.

## Stop rule

Do not:
- mine additional handcrafted source phases;
- use H32 to choose a local phase;
- add a local alignment regularizer;
- add a phase-specific semantic loss.

The short-horizon semantic-target mining branch is closed.

## Next research-level question

If semantic repair is still required, the next branch should move away from local/phase-specific differential targets.

The defensible alternatives are higher-level:
- trajectory-level preference-outcome consistency;
- return-level semantic constraints;
- sequence-level credit representation;
- or a training formulation that directly optimizes preference-conditioned multi-step semantic outcomes.

Any such branch should be treated as a new methodology question, not as a continuation of local gradient repair.
