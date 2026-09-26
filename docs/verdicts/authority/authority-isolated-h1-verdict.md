# Authority-Isolated AI-H1 Verdict

Status: **FROZEN — AI-H1 FAIL — FOUNDATION; AI-H2 NOT AUTHORIZED**
Date: 2026-09-25

## Formal result
The sole-path authority hypothesis does **not** fail by authority drift.

Authority increases strongly through the screen:

    pairwise preference separation
      u0   0.0475
      u10  0.1000
      u25  0.1770
      u50  0.1770
      u75  0.4914

    simplex-tangent Jacobian norm
      u0   0.1221
      u10  0.2655
      u25  0.4790
      u50  0.4430
      u75  1.3258

Both frozen authority-retention criteria pass by a large margin.
## Authority retention
At u75:

    pairwise / prior peak   = 2.776
    pairwise / u0           = 10.356
    Jacobian / prior peak   = 2.768
    Jacobian / u0           = 10.855

Thus the behavior seen in V2-PF — meaningful authority followed by loss of final authority — is not reproduced when the actor has only one preference pathway.

This is positive mechanistic evidence for the authority-allocation / pathway-competition diagnosis.

## Parameter-family geometry
The generated parameter family remains strongly non-degenerate at u75:

    centered s2/s1              0.817
    centered effective rank     3
    largest direction energy    0.552
    parameter specific fraction 0.269

The simplex-tangent parameter-manifold Jacobian is also nonzero:

    Frobenius norm               2.889
## Functional geometry
Preference-conditioned functional geometry becomes increasingly distinct:

    functional effective rank    4
    functional s2/s1             0.472

However the predeclared functional specific-energy fraction is:

    0.108 < 0.15

so the frozen functional-geometry criterion fails.

Because the common RMS includes the large state-dependent action component, this metric is conservative for a sole-path actor, but the threshold was frozen and is not changed post hoc.

## Foundation trajectory
Foundation checks are healthy through u50:

    u0   early EV 0.526   late EV 0.211
    u10  early EV 0.543   late EV 0.301
    u25  early EV 0.528   late EV 0.203
    u50  early EV 0.433   late EV 0.314

At u75 the late critic collapses:

    early EV                 0.581
    late EV                 -1.382
    late negative fraction   0.45
    combined negative frac   0.25
Other guardrails remain valid:

    max PPO ratio error          6.10e-5
    last-10 termination fraction 0.025
    family output gradients      observed
    family basis gradients       observed

Therefore the final H1 failure classification is:

> **AI-H1 FAIL — FOUNDATION**

not authority drift.

## Interpretation
The experiment isolates the newest hypothesis cleanly.

V2-PF had two actor-side preference routes and showed authority peaking mid-training then declining.

The authority-isolated actor has one preference route only and shows the opposite behavior: preference authority remains present and grows substantially through u75.

This supports the claim that functional redundancy / pathway competition contributed to V2-PF authority drift.

However, the final policy also moves into a regime where the validated late critic no longer generalizes adequately. Because actor updates depend on that critic, semantic evaluation at u75 would confound authority isolation with critic-support failure.
## Decision
AI-H2 semantic evaluation is **not authorized**.

Do not claim that authority isolation solves semantic forgetting.

Do not relax the critic gate or select u50 post hoc as the final model.

The next technical question, if this branch is continued, is not pathway ownership anymore. It is whether the validated critic/support mechanism can remain valid under the much stronger sole-path preference authority without changing the actor treatment.

That would require a separately predeclared foundation-repair/compatibility audit, not an H2 semantic run.

## Thesis wording
> Removing the competing shared preference pathways changed the authority dynamics qualitatively. In the authority-isolated actor, pairwise preference separation and simplex-tangent preference sensitivity increased rather than decayed, reaching 10.4× and 10.9× their transferred initial values at the final checkpoint. The generated parameter family remained multidimensional. This contrasts with V2-PF, where family-path authority peaked mid-training and declined by the final H1 checkpoint, providing evidence that pathway competition contributed to authority drift. However, the authority-isolated screen did not pass its full validation contract because late critic explained variance collapsed to -1.38 at the final checkpoint. Semantic evaluation was therefore blocked: stable preference authority was demonstrated, but its effect on semantic retention remains unresolved under a valid critic foundation.
