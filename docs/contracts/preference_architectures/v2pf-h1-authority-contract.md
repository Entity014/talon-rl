# V2-PF H1 Policy-Family Authority / Non-Degeneracy Contract

Status: **PREDECLARED — SEMANTICS BLOCKED**
Date: 2026-09-25

## Purpose
Test whether the one-model V2-PF architecture learns a genuinely preference-indexed policy family rather than a larger common residual.

## Training scope
- Start from frozen H0 checkpoint `v2pf_h0_init.pt`.
- One jointly trained model only.
- Same V2-H1 architecture-screen semantics for direct comparison.
- 75 actor updates; snapshots at 0, 10, 25, 50, 75.
- PPO objective, Foundation V2 critic refresh/support, GAE lambda=.95, preference batching, survival checks unchanged.
- No semantic endpoint judgement in H1.

## Primary geometry
For T/A/O/S/C generated parameter vectors:
`theta_pf(w) = vec(Delta W1, Delta b1, Delta W2, Delta b2)`.

Define common and centered realizations:
`theta_bar = mean_w theta_pf(w)`
`theta_tilde(w) = theta_pf(w) - theta_bar`.

Primary parameter-family metrics:
- pairwise centered parameter distance;
- centered SVD singular spectrum;
- centered effective rank at 5% of s1;
- centered s2/s1;
- centered specific-energy fraction
  `E_spec/(E_common+E_spec)`,
  where `E_common=||theta_bar||` and
  `E_spec=sqrt(mean_w ||theta_tilde(w)||^2)`.

## Functional family residual
On the frozen H0 probe states:
`Delta a_PF(s,w)=a_full(s,w)-a_PFmasked(s,w)`.

Center across T/A/O/S/C at each state and report:
- pairwise centered residual distance;
- centered residual SVD / effective rank;
- functional specific-energy fraction;
- pairwise action separation with/without family block;
- preference-Jacobian norm with/without family block.

## Frozen H1 PASS criteria
All must hold at update 75:

1. coefficient separation:
   mean heavy-preference pairwise ||c(w_i)-c(w_j)|| >= 0.02.
2. generated parameter separation:
   mean heavy pairwise ||theta_pf(w_i)-theta_pf(w_j)|| >= 0.01.
3. centered parameter manifold:
   s2/s1 >= 0.10 AND effective rank >= 2.
4. centered parameter specific-energy fraction >= 0.15.
5. centered functional residual:
   effective rank >= 2 AND specific-energy fraction >= 0.15.
6. functional authority:
   mean ||Delta a_PF|| >= 1e-3.
7. family-block masking must reduce either:
   - pairwise preference action separation by >=5%, OR
   - preference-Jacobian Frobenius norm by >=5%.
8. preference-specific functional residual must be nontrivial:
   centered residual RMS >= 25% of common residual RMS.
9. coefficient and generated-parameter Jacobians with respect to w are nonzero:
   coefficient Jacobian Fro norm >1e-3;
   parameter Jacobian Fro norm >1e-3.
10. centered parameter geometry must not regress to the V2-H common-direction failure:
    heavy centered effective rank >=2 and no single centered singular direction explains >90% centered energy.
11. critic/foundation:
    early EV >0, late EV >0;
    each phase negative-EV fraction <=.25;
    combined negative fraction <=.25.
12. PPO/safety:
    max pre-update ratio error <=1e-4;
    last-10 mean termination fraction <.5.
13. learning-path activity:
    coefficient-output gradient and full-rank basis gradients are observed above 1e-7.

## Verdict
- **V2-PF H1 PASS**: all criteria pass -> H2 semantic accumulation authorized.
- **V2-PF H1 FAIL — COMMON FAMILY COLLAPSE**: geometry/authority criteria fail -> close policy-family escalation; do not tune K/depth/basis count.
- **V2-PF H1 FAIL — FOUNDATION**: critic/PPO/safety criteria fail -> architecture screen invalid; no H2.

No H1 metric may be re-thresholded after observing results.
