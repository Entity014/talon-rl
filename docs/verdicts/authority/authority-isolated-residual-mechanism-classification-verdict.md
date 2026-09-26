# Authority-Isolated Residual Mechanism Classification Verdict

Status: **FROZEN — READ-ONLY CLASSIFICATION COMPLETE**
Date: 2026-09-25

## Scope

No additional training was performed.

The remaining unclassified residuals were evaluated with the same donor/phase-basin protocol:

- single-coordinate donor matrix
- grouped-coordinate donor matrix
- phase-window donor
- donor-strength interpolation
- aligned state/action divergence

Cases:

- suite2 / O / lane2
- suite3 / S / lane0
- suite3 / C / lane0

Previously established classes:

- Class A: suite2/O/lane7 localized FL-hip regression, critical around t6-10
- Class B: suite3/O/lane0 phase-sensitive hip/dynamics basin failure

---

# Result: residuals collapse into three mechanism classes

## Class A — localized FL-hip regression

Case:

    suite2 / O / lane7

Properties:

- repaired-u10 fails, frozen u75 survives
- u75 FL_hip alone rescues
- other single-coordinate donors do not
- critical donor window is narrow: t6-10
- pattern does not reproduce across other O-heavy residuals

Interpretation:

    localized training regression
    not a generic actuator method

---

## Class B — suite3 mid-phase dynamics-basin family

Cases:

    suite3 / O / lane0
    suite3 / S / lane0
    suite3 / C / lane0

Shared evidence:

1. robust u50 survives all three
2. u75 fails all three
3. repaired-u10 fails all three
4. alternate hip/support geometry can rescue
5. rescue is strongly phase-sensitive
6. donor-strength response is non-monotonic or thresholded
7. the relevant intervention happens before terminal contact

### suite3/O/lane0

Previously established:

- u50 FL_hip alone rescues
- u50 RR_hip alone rescues
- full-u50 donor windows t4-8, t6-10, t9-13 rescue
- FL_hip windows t6-10 or t9-13 rescue
- RR_hip windows t4-8 or t6-10 rescue
- donor-strength response is non-monotonic

### suite3/S/lane0

Single-coordinate u50 rescuers:

    RR_hip
    FR_thigh
    FL_calf

Single-coordinate u75 rescuers:

    FL_hip
    RR_hip

Group u50 rescuers include:

    hips
    front-right
    rear-right
    rear
    whole policy

Representative phase localization:

    u50 RR_hip      t4-8, t6-10, t9-13, t11-15 rescue
    u50 FR_thigh    t9-13 rescue
    u50 FL_calf     t0-5 rescue
    u75 FL_hip      t6-10, t9-13 rescue
    u75 RR_hip      t4-8, t6-10 rescue

Strength behavior is not a simple scalar margin:

    u50 RR_hip      alpha .75, 1.0 rescue
    u50 FR_thigh    alpha .50, 1.0 rescue; .75 fails
    u75 whole       alpha .25, .50 rescue; .75, 1.0 fail

### suite3/C/lane0

Single-coordinate u50 rescuers:

    FL_hip
    RR_hip
    FR_thigh

Single-coordinate u75 rescuers:

    FL_hip
    RR_hip

Group topology closely matches suite3/S.

Representative phase localization:

    u50 FL_hip      t4-8, t6-10 rescue
    u50 RR_hip      t4-8, t6-10, t11-15 rescue
    u50 FR_thigh    t9-13, t11-15 rescue
    u75 FL_hip      t6-10, t9-13 rescue
    u75 RR_hip      t4-8, t6-10 rescue

Strength response again shows basin geometry:

    u50 FL_hip      alpha .75, 1.0 rescue
    u50 RR_hip      alpha .25, .75, 1.0 rescue; .50 fails
    u75 whole       alpha .75 rescues while other tested strengths fail

### Aligned divergence

For suite3/S and suite3/C, repaired-u10 stays dynamically close to u75 for much of the trajectory.

Relative to u75:

- action differences appear early
- angular-state divergence does not become clearly large until roughly t12
- roll/pitch divergence remains small until immediately before contact
- both cases show a large state split around t15 after the critical mid-phase action geometry has already been set

This is consistent with delayed closed-loop basin separation rather than an immediate action-magnitude failure.

### Class B interpretation

suite3 O/S/C lane0 are one recurring mechanism family:

    mid-phase multi-actuator controller geometry
        ->
    delayed dynamics-basin selection
        ->
    high angular motion / base contact

The exact rescuing coordinate differs by preference, but the causal structure recurs across three residual cases.

This class therefore satisfies the recurrence criterion better than Class A and is the strongest candidate for a future targeted robustness formulation.

---

## Class C — suite2/O/lane2 distributed O-heavy basin residual

Case:

    suite2 / O / lane2

Baseline:

    u50        survives
    u75        fails @15
    repaired   fails @15

Single-coordinate u50 rescuers:

    FL_hip
    RR_hip
    FR_thigh

Many u50 groups rescue:

    hips
    front-left
    front-right
    rear-right
    front
    rear
    whole policy

No tested single-coordinate u75 donor rescues.

No tested u75 group donor rescues.

However, whole-u75 donor is itself non-monotonic:

    alpha .10      rescues
    alpha .25+     fails

u50 whole-policy donor:

    alpha .10      fails
    alpha .25+     rescues

Phase behavior is unusually broad:

    full u50 donor t0-5      rescues
    full u50 donor t4-8      rescues
    full u50 donor t6-10     rescues
    full u50 donor t9-13     rescues
    full u50 donor t11-15    rescues

Individual rescuers have different windows:

    u50 FL_hip      t6-10
    u50 RR_hip      t4-8, t6-10
    u50 FR_thigh    every tested window

Aligned divergence:

- repaired vs u75 action difference appears early
- orientation-state divergence remains relatively small until later
- angular-velocity divergence becomes substantial around t10
- by t14 the repaired/u75 trajectories differ dynamically but both are in the failing basin

### Class C interpretation

This does not match Class A:

- rescue is not FL-hip-specific
- multiple coordinates and broad groups rescue

It also does not match Class B exactly:

- full-u50 rescue is not confined to a mid-phase window
- FR_thigh donor rescues across every tested phase
- the rescue topology is substantially broader and more distributed

Best current description:

    distributed O-heavy controller basin / policy-geometry residual

This class currently has only one residual case and therefore does **not** meet the recurrence criterion for authorizing a new training method by itself.

---

# Mechanism clustering summary

The five repaired-u10 residual failures now cluster as:

    Class A
      suite2/O/lane7
      localized FL-hip t6-10 regression
      n = 1

    Class B
      suite3/O/lane0
      suite3/S/lane0
      suite3/C/lane0
      recurring mid-phase dynamics-basin family
      n = 3

    Class C
      suite2/O/lane2
      distributed O-heavy controller-basin residual
      n = 1

Thus only Class B currently recurs across multiple residual cases.

---

# Stop-rule decision

The predeclared recurrence requirement is satisfied only for Class B.

Therefore:

- do not train a Class-A FL-hip patch
- do not train a Class-C lane2-specific patch
- do not resume global headroom tuning
- do not hard-code reset-specific corrections

If training is reopened, it should target the **Class-B recurring mid-phase dynamics-basin mechanism**, not individual lanes.

The method should remain generic across O/S/C and should not encode a specific preference label or a specific reset lane.

AI-C2 and AI-H2 remain blocked until a Class-B-targeted repair passes:

    authority retention
    + wide critic validity
    + suites2/3 min survival >= .95
    + no new Class-A/Class-C regressions
