# AI-C0/C1 Critic-Only Capacity Verdict

Status: **FROZEN — CRITIC CAPACITY SUPPORTED, FORMAL AI-C1 FAILS SURVIVAL; AI-C2 NOT AUTHORIZED**
Date: 2026-09-25

## AI-C0
Actor invariance passes exactly:

    deterministic action max error          0
    stochastic/pre-tanh/log-prob max error 0

Only the critic representation changed.

## Fixed-policy critic result
The widened critic (52→256→256→128→4) was trained only on independent current-policy support with the AI-H1 u75 actor frozen.

Held-out dense resets:

    old critic H32 EV       0.401
    wide critic H32 EV      0.491

    old critic MC64 EV      0.425
    wide critic MC64 EV     0.444

    wide H32 negative frac  0.025
    wide MC64 negative frac 0.025

Semantic evaluation suites:

    old critic H32 EV       0.485
    wide critic H32 EV      0.536

    old critic MC64 EV      0.453
    wide critic MC64 EV     0.455

    wide H32 negative frac  0.000
    wide MC64 negative frac 0.0125

The two previously problematic value heads are recovered:

    Angular H32 EV          0.807
    Smoothness H32 EV       0.611

Thus the minimal critic-capacity escalation successfully restores fixed-policy value generalization without changing actor behavior.

## Why the formal gate still fails
The predeclared AI-C1 contract also required:

    min survival >= 0.95

on both held-out dense and semantic suites.

Held-out dense survival:

    1.000  PASS

Semantic-suite minimum survival:

    0.875  FAIL

This survival result is identical for old-critic control and widened-critic treatment because the actor is frozen.

The failure is therefore not caused by critic replacement.

## Survival localization
On the frozen AI-H1 u75 actor:

    suite 1: T/A/O/S/C = 1.000

    suite 2:
      T = 1.000
      A/O/S/C = 0.875

    suite 3:
      T/A/O/S/C = 0.875

The issue is therefore broader than a single preference outlier and indicates that the u75 high-authority actor itself has entered a less robust regime on some frozen semantic reset suites.

## Interpretation
Two conclusions must remain separate:

1. **Critic representation compatibility is repairable.**
   A modest width increase is sufficient to recover H32/MC64 generalization, including Angular and Smoothness, on unseen current-policy support and semantic suites.

2. **The full u75 branch is not yet validated for continuation.**
   The frozen actor misses the predeclared survival floor on some semantic suites, so a paired AI-C2 RL rerun or AI-H2 semantic claim would still be confounded.

The earlier diagnosis that support-only repair was insufficient remains valid; what was needed on the critic side was additional representation capacity, not more head refresh alone.

## Decision
Formal verdict:

> **AI-C1 FAIL — FROZEN ACTOR SURVIVAL GATE**

but mechanistic critic verdict:

> **MINIMAL CRITIC CAPACITY REPAIR SUCCEEDS UNDER FIXED POLICY**

AI-C2 is not authorized under the frozen contract.

Do not relax the survival threshold post hoc.
Do not reduce actor authority merely to make the gate pass.
Do not claim that authority isolation already solves semantic forgetting.

The next unresolved issue is whether the high-authority u75 actor's reduced survival is a genuine consequence of the sole-path formulation/training dynamics or a boundary/reset-specific robustness problem.

## Thesis wording
> A critic-only capacity escalation from a 128-wide shared body to a 256–256–128 body restored fixed-policy value generalization under the authority-isolated u75 policy, including positive H32 values for the previously failing Angular and Smoothness heads. However, the predeclared C1 gate still failed because the frozen actor achieved only 0.875 minimum survival on some semantic reset suites. Since survival was identical for the old- and widened-critic controls, this residual blocker belongs to the actor's high-authority policy regime rather than to critic representation.
