# Teacher V4 — F7 Dynamic-Stability Semantic Definition (stage 1: phenotype map)

Status: **FROZEN 2026-09-28 before computation. Descriptive, read-only, no gates. Nothing is trained or selected.**
Branch: `v4-c2-semantic-preservation`
Follows: [F6 verdict](../../verdicts/teacher_v4/teacher-v4-f6-conditional-feature-verdict.md)

## Why

F6 found that the current D realization, A = ‖ω_xy‖², can be made small by
moving the body's dynamics into the vertical axis (a bouncing gait with
flight phases), which A does not see. We have not operationally defined
what Dynamic Stability should cover. F7 separates the **semantic target**
(physical behavior) from the **raw reward candidates**, and defines the
target without using any candidate reward to build it.

## Stage 1 (this contract): phenotype map

A vector of physical observables per behavior, **never a weighted sum** (a
sum would need coefficients chosen before selection):

| component | observable (per 32-step window of steps 33–128) |
|---|---|
| rotational dynamics | RMS ‖ω_xy‖ |
| vertical dynamics | RMS \|v_z\| |
| vertical excursion | peak-to-peak of the linearly detrended ∫ v_z dt (body-frame v_z, approximate) |
| contact continuity | flight fraction (all four feet off the ground) |
| orientation guardrail | mean tilt. Reported only; **not** a D component, because tilt is O. |

Behaviors, all from existing F6 traces (selection data): s73102 C and T⁺
(low-A), F4 s74102 / s74103 T⁺ (high-A), M0, standing (s73101 C), F4 s74102
/ s74103 C. Output: median and 10–90 % range per component.

## What stage 1 can and cannot conclude

It shows whether the existing behaviors order consistently along one axis,
or whether the components disagree (for example, low rotational but high
vertical dynamics). It does not choose between:

1. A + lin_vel_z as constituents of one D;
2. vertical stability as a constraint, not a preference axis;
3. D split into rotational and vertical objectives;
4. A as the correct D, with vertical bounce an acceptable gait.

That choice is a semantic decision, recorded separately before any new
feature selection. The next stage (supervised feature/role screen against
the chosen target) and the acquisition of more walkers (b) each need their
own contract.

Script: `f7_phenotype.py`. Output: `runs/teacher_v4_f7-2026-09-28/f7_phenotype.json`.
