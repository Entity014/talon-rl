# Authority-Isolated AI-H0/H1 Contract

Status: PREDECLARED — SEMANTICS BLOCKED
Date: 2026-09-25

## AI-H0
Start from transferred `student_step_5000.pt`. No RL updates. Require:
- held-out transfer bounds unchanged from feasibility audit;
- simplex-tangent Jacobian rel error <=15%, cosine >=.95;
- actor graph has no direct/embed/FiLM preference path;
- critic output matches frozen V2-B within 1e-6;
- deterministic/stochastic action and log-prob are finite; pre-update PPO ratio error <=1e-6;
- checkpoint roundtrip action/value error <=1e-6;
- no-update rollout survival >=.95 over frozen reset suites.
AI-H1 blocked unless all pass.

## AI-H1
75 RL actor updates with exact V2-H1/V2-PF H1 budget and preference schedule; Foundation V2 critic refresh/support and lambda=.95 unchanged. Snapshots 0/10/25/50/75. No semantic judgement.

Primary sole-path authority metrics at each snapshot:
1. mean pairwise T/A/O/S/C action separation on frozen probe states;
2. simplex-tangent action-Jacobian Frobenius norm at center;
3. generated parameter-family centered geometry;
4. centered functional-family geometry.

Authority retention at u75 is measured against peak over u10/u25/u50:
- pairwise-action retention ratio >=0.75;
- tangent-Jacobian retention ratio >=0.75.
Also require u75 >=0.75 of u0 transferred authority for both metrics.

Family geometry at u75 must retain:
- centered parameter effective rank >=2; s2/s1 >=.10; specific fraction >=.15;
- centered functional effective rank >=2; specific fraction >=.15;
- coefficient and generated-parameter Jacobians wrt simplex preference nonzero (>1e-3 Fro norm).

Foundation guardrails:
- early and late critic EV >0; each phase negative fraction <=.25; combined <=.25;
- max pre-update PPO ratio error <=1e-4;
- last-10 termination fraction <.5;
- family hyper and basis gradients observed >1e-7.

Verdicts:
- AI-H1 PASS: all criteria pass -> AI-H2 semantic gate authorized.
- AI-H1 FAIL — AUTHORITY DRIFT: authority retention fails despite valid foundation.
- AI-H1 FAIL — FAMILY DEGENERATION: family geometry fails despite valid foundation.
- AI-H1 FAIL — FOUNDATION: PPO/critic/safety guard fails.

No threshold or update budget may be changed after observing results.
