# Pre-Contact Common/Family Compatibility Audit Verdict

Status: **FROZEN — INTERACTION OWNERSHIP CONFIRMED, BUT NO TRANSFERABLE GENERIC COMPATIBILITY INVARIANT FOUND**
Date: 2026-09-25

## Question

For interaction-only fresh u40 failures, is there a recurrent common/family action-generation incompatibility on matched pre-contact states that is sufficiently generic to justify a new training-side compatibility constraint?

## Cases

Interaction-only cases from the u30/u40 path-swap audit:

NARROW:
- T / seed 9700226 / lane6

WIDE:
- T / seed 9700226 / lane6
- A / seed 9701226 / lane5
- S / seed 9703226 / lane3
- C / seed 9704339 / lane0

For each case, use the full-u40 trajectory and retain the final 10 pre-contact states.

Evaluate the same states through the 2x2 actor matrix:

    C30F30
    C40F30
    C30F40
    C40F40

This removes visitation confounding.

## Interaction definition

Pre-tanh:

    I_z =
      z_40,40
      - z_40,30
      - z_30,40
      + z_30,30

Final action:

    I_a =
      a_40,40
      - a_40,30
      - a_30,40
      + a_30,30

## Aggregate interaction

Across 5 cases x 10 pre-contact states:

Mean interaction norm by case:

    narrow T/9700226:
      mean ||I_z||  .067
      mean ||I_a||  .259

    wide T/9700226:
      mean ||I_z||  .102
      mean ||I_a||  .342

    wide A/9701226:
      mean ||I_z||  .227
      mean ||I_a||  .456

    wide S/9703226:
      mean ||I_z||  .157
      mean ||I_a||  .606

    wide C/9704339:
      mean ||I_z||  .212
      mean ||I_a||  .331

Thus interaction is functionally real at the action level.

## Action coordinate mapping

    0  FL_hip
    1  FR_hip
    2  RL_hip
    3  RR_hip
    4  FL_thigh
    5  FR_thigh
    6  RL_thigh
    7  RR_thigh
    8  FL_calf
    9  FR_calf
    10 RL_calf
    11 RR_calf

## Raw recurrence

Pre-tanh top-3 recurrence:
- FR_hip: 5/5 cases
- RR_thigh: 4/5
- FL_hip: 2/5
- RL_hip: 2/5

Action-space top-3 recurrence:
- RL_hip: 5/5
- RR_hip: 3/5
- FR_calf: 3/5

This initially appears to suggest a shared compatibility pattern.

## Normalized interaction

However raw recurrence is confounded by coordinates whose single-path response is naturally small or large.

Normalize interaction magnitude by the magnitude of the two single-path changes.

Pre-tanh:

    median ||I_z|| / (||Delta_common|| + ||Delta_family||)

ranges only approximately:

    .015 - .046

across cases.

Therefore the logit-level common/family non-additivity is small.

After tanh/output realization:

    median ||I_a|| / (||Delta_common|| + ||Delta_family||)

ranges approximately:

    .166 - .874

and is strongly case-dependent.

The output nonlinearity therefore amplifies small path interactions in some state/action coordinates.

## Last-5-step recurrence

The final 5 pre-contact steps were analyzed separately.

### RL_hip

Absolute action interaction is large in every case:

    mean across cases     .308
    minimum case mean     .178

Normalized interaction is also substantial:

    mean normalized ratio 1.10

But sign is not stable across all cases:
- narrow T: mixed
- wide T: mixed
- wide A: mixed
- wide S: consistently negative
- wide C: consistently negative

Therefore RL_hip is recurrent in magnitude, but not in directional compatibility signature.

### RR_hip

    mean absolute interaction   .141
    mean normalized ratio      1.30
    global sign mean          +.84

Sign is comparatively consistent.

However magnitude is highly heterogeneous:

    minimum case mean .014
    maximum case mean .455

Thus it is not a uniformly active mechanism.

### FR_calf

    mean absolute interaction   .107
    mean normalized ratio       .367
    global sign mean           -.96

Direction is highly consistent.

But magnitude ranges from nearly zero in some cases to .379 in another.

Therefore sign recurrence alone is insufficient.

### RR_thigh

Normalized interaction is high and sign-consistent:

    mean normalized ratio 1.40
    global sign mean      -.92

But absolute action interaction is tiny:

    mean .013
    minimum .0003

The high normalized score is largely caused by near-zero single-path action deltas.

RR_thigh is therefore a normalization artifact rather than a robust causal coordinate.

## Key interpretation

The interaction-only failures are real:

    full u40 fails
    both single-path hybrids survive

and same-state action generation contains measurable common/family interaction.

However the recurrent coordinate signatures do not jointly satisfy:

    large absolute effect
    +
    large normalized effect
    +
    stable sign
    +
    stable phase
    +
    recurrence across T/A/S/C

No coordinate or morphology group meets all conditions.

What recurs is a broader phenomenon:

    small common/family pre-tanh mismatch
      ->
    state/coordinate-dependent nonlinear output amplification
      ->
    different action coordinates become critical in different cases

This is an interaction phenomenon, but not a single transferable compatibility invariant.

## Decision

Confirmed:

    common/family interaction ownership       YES
    same-state action interaction             YES
    output nonlinearity amplifies interaction YES

Not confirmed:

    one recurrent joint mechanism             NO
    one recurrent sign/magnitude signature    NO
    one morphology-level compatibility rule   NO
    generic compatibility regularizer target  NO

Therefore:

    compatibility regularization              NOT AUTHORIZED
    joint-specific penalty                     NOT AUTHORIZED
    new tanh/saturation penalty                NOT AUTHORIZED
    further coordinate mining                  STOP
    AI-H2                                      REMAINS BLOCKED

## Stop-rule conclusion

The predeclared stop rule is met:

    interaction signatures differ too much across cases
    to support a generic training method.

Do not continue searching for additional handcrafted interaction features.

The current fresh-u40 failures are best treated as heterogeneous late-continuation robustness regressions rather than a single missing mechanistic regularizer.

## Implication for the project

The strongest generalizable result remains:

1. preference authority ownership was repaired;
2. wide critic value representation was repaired;
3. robustness at u20/u30 was achieved;
4. authority durability was repaired with a lower simplex-edge floor;
5. continued u30->u40 optimization reintroduces heterogeneous robustness failures;
6. those failures are not explained by:
   - authority contraction,
   - global over-expansion,
   - saturation,
   - common path alone,
   - family path alone,
   - or one reusable common/family compatibility invariant.

The defensible next engineering choice is therefore not another mechanism-specific loss.

Candidate next strategy:

    treat u30 as the validated operating checkpoint
    and evaluate the semantic forgetting question there,
    explicitly as a checkpoint-level test rather than claiming indefinite training stability.

If the research contract absolutely requires continued-learning stability before AI-H2, then the next branch must be framed as generic late-training stabilization / early-stopping / checkpoint-selection methodology, not further mechanism mining.
