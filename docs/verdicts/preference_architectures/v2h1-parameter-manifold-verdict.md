# V2-H1 Parameter-Manifold Authority Verdict

Status: **FROZEN — V2-H1 FAIL; V2-H2 NOT AUTHORIZED; V2-H BRANCH CLOSED**

Date: 2026-09-24

## Question

Does the preference-conditioned hypernetwork learn a non-degenerate, preference-dependent policy-parameter manifold with causal influence on action?

V2-H1 intentionally does **not** evaluate semantic endpoint success.

## Frozen contract

- V2-H0 exact function-preserving initialization
- Foundation V2
- complete V2-B direct / embedding / FiLM path
- critic architecture and support protocol
- GAE lambda = 0.95
- repaired PPO/action-logprob semantics
- objective definitions
- seed 73001
- 75-update authority screen
- no semantic threshold changes
- no basis-count, rank, coefficient-network, learning-rate, lambda, entropy, or foundation tuning

Treatment only:

    c(w) = HyperNet(w)
    DeltaW(w) = sum_k c_k(w) U_k V_k
    W_actor(w) = W0 + DeltaW(w)

with four unlabeled rank-4 latent weight bases on the actor action head.

## Predeclared H1 gate

H1 required all of the following:

1. Preference -> coefficient authority
   - heavy-preference mean pairwise coefficient distance >= 0.02
   - coefficient manifold s2/s1 >= 0.05
   - coefficient Jacobian Frobenius norm > 1e-3

2. Coefficient -> policy-parameter authority
   - heavy-preference mean pairwise DeltaW Frobenius distance >= 1e-4
   - mean pairwise (1 - cosine) of generated DeltaW directions >= 0.02
   - generated-delta manifold s2/s1 >= 0.05
   - local parameter-manifold Jacobian s2/s1 >= 0.05
   - parameter Jacobian Frobenius norm > 1e-4

3. Generated parameters -> action authority
   - hyper-path action authority > 1e-4
   - masking DeltaW reduces action separation or action-preference Jacobian by > 1e-4

4. Trainability / foundation
   - coefficient-output gradient observed
   - basis gradient observed
   - Foundation-V2 critic / PPO-ratio / survival guardrails remain valid

All criteria were predeclared before training.

## Result

### 1. Preference -> coefficient authority — PASS

The coefficient network does respond to preference.

Final coefficient vectors:

| Preference | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|
| T-heavy | -0.07838 | -0.03462 | -0.06508 | 0.07895 |
| A-heavy | -0.07258 | -0.05121 | -0.06796 | 0.09813 |
| O-heavy | -0.08039 | -0.03299 | -0.08022 | 0.08101 |
| S-heavy | -0.05656 | -0.02658 | -0.05810 | 0.06263 |
| Center | -0.07261 | -0.03696 | -0.06836 | 0.08126 |

Heavy-preference coefficient geometry:
- mean pairwise distance = **0.03081**
- threshold = **0.02**
- minimum pairwise distance = **0.01549**
- maximum pairwise distance = **0.04712**

Centered coefficient singular values:
- s1 = **0.03391**
- s2 = **0.01899**
- s3 = **0.00763**
- s2/s1 = **0.5600**
- effective rank at 5% threshold = **3**

Local coefficient Jacobian at center:
- Frobenius norm = **0.07182**
- singular values = **[0.05919, 0.03849, 0.01310, 0.00113]**
- s2/s1 = **0.6502**
- effective rank = **3**

Thus the hypernetwork does not collapse to a single coefficient vector or a rank-1 coefficient map.

### 2. Preference -> generated DeltaW magnitude / rank — PASS

Generated action-head parameter deltas differ measurably across preferences.

Heavy-preference pairwise Frobenius distance:
- mean = **9.462e-4**
- minimum = **4.340e-4**
- maximum = **1.551e-3**
- threshold = **1e-4**

Centered generated-delta singular values:
- s1 = **1.163e-3**
- s2 = **4.445e-4**
- s3 = **1.499e-4**
- s2/s1 = **0.3822**
- effective rank = **3**

Local parameter-manifold Jacobian:
- Frobenius norm = **0.002196**
- singular values = **[0.001974, 0.000926, 0.000257, 0.000024]**
- s2/s1 = **0.4690**
- effective rank = **3**

Therefore the learned mapping has genuine local multi-dimensional variation in parameter space.

### 3. Absolute generated-parameter direction diversity — FAIL

This is the decisive failed criterion.

Cosine matrix of the absolute generated DeltaW vectors:

| | T | A | O | S |
|---|---:|---:|---:|---:|
| T | 1.0000 | 0.9953 | 0.9988 | 0.9990 |
| A | 0.9953 | 1.0000 | 0.9924 | 0.9954 |
| O | 0.9988 | 0.9924 | 1.0000 | 0.9996 |
| S | 0.9990 | 0.9954 | 0.9996 | 1.0000 |

Direction-diversity statistic:
- mean pairwise (1 - cosine) = **0.00323**
- predeclared threshold = **0.02**
- minimum = **0.00043**
- maximum = **0.00757**

Although the *differences around the mean* span approximately three directions, the full generated parameter deltas remain dominated by a common direction.

This is exactly the failure mode the cosine gate was designed to detect:

> coefficient vectors differ and centered parameter variation has nontrivial rank, but all preferences still receive nearly the same dominant adaptive weight displacement.

The resulting object is therefore closer to:

> shared adaptive action-head residual + small preference-dependent perturbations

than to:

> clearly distinct preference-indexed policy-parameter realizations.

### 4. Basis contributions — preference-dependent but dominated by similar mixture structure

Basis-contribution shares:

| Preference | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| T-heavy | 0.2899 | 0.0953 | 0.2403 | 0.3746 |
| A-heavy | 0.2384 | 0.1252 | 0.2228 | 0.4135 |
| O-heavy | 0.2782 | 0.0850 | 0.2771 | 0.3597 |
| S-heavy | 0.2635 | 0.0921 | 0.2702 | 0.3742 |
| Center | 0.2664 | 0.1009 | 0.2503 | 0.3824 |

The mixture changes by preference, but not enough to rotate the generated policy delta into clearly distinct parameter directions.

### 5. Generated parameters -> action authority — PASS, but small

The hyper-path is causally active.

At the final checkpoint:
- mean ||a_full - a_masked|| = **0.002332**
- threshold = **1e-4**

Preference-conditioned action separation:
- full V2-H = **0.059033**
- hyper-path masked = **0.058761**
- increase from hyper path = **0.000271**

This exceeds the predeclared 1e-4 masking criterion.

Action-preference Jacobian:
- full V2-H = **0.032101**
- hyper-path masked = **0.032073**

The Jacobian increment is small, but the action-separation criterion independently passes.

Therefore the generated weights are not merely latent decoration; they have measurable causal influence on action.

### 6. Hypernetwork and bases are trainable — PASS

Gradient diagnostics:
- max coefficient-output gradient norm = **0.03781**
- median coefficient-output gradient norm = **0.00336**
- max hyper-U gradient norm = **0.03956**
- max hyper-V gradient norm = **0.02304**

Both coefficient mapping and low-rank bases receive training signal.

The final basis norms also change from H0:
- hyper-U norm: 0.2000 -> **0.2619**
- hyper-V norm: 0.4000 -> **0.6795**

Thus the direction-collapse result is not caused by disconnected or frozen hypernetwork parameters.

### 7. Foundation V2 guardrails — PASS

Final critic audit:
- early EV mean = **0.3564**
- early negative fraction = **0.15**
- late EV mean = **0.2682**
- late negative fraction = **0.10**
- combined negative fraction = **0.125**

Optimization / survival:
- max PPO ratio error = **3.05e-5**
- last-10-update termination fraction = **0.0**

Therefore H1 fails because of policy-manifold direction diversity, not because the training foundation destabilized.

## Interpretation

V2-H succeeds at several things that V2-C did not:

- preference maps to distinct latent coordinates;
- generated parameters differ by preference;
- the local parameter manifold has approximately three meaningful directions;
- the generated parameter path causally affects action.

However, the absolute generated policy deltas remain nearly parallel across T/A/O/S-heavy preferences.

This means V2-H did not satisfy the stronger hypothesis required by the branch:

> preferences should produce sufficiently distinct policy-parameter realizations rather than a common adaptive parameter shift with small preference-dependent deviations.

The distinction between centered rank and absolute direction is important.

A rank-3 centered cloud does **not** imply that the generated policies occupy substantially different parameter directions if a large common component dominates every point.

The predeclared cosine criterion was specifically included to reject this case.

## Decision

**V2-H1 FAIL.**

Per the frozen stop rule:

- V2-H2 semantic accumulation gate is **NOT AUTHORIZED**
- no semantic endpoint judgment is made from V2-H1
- no generated-weight retention is introduced
- no basis-count tuning
- no rank tuning
- no coefficient-network-depth tuning
- no common-component subtraction
- no orthogonality/diversity regularizer
- no learning-rate / lambda / entropy tuning

The V2-H branch is closed at the parameter-manifold authority stage.

The final thesis reference returns to:

> **V2-B + Foundation V2 + GAE lambda = 0.95 + no retention intervention**

## Scientific implication

This result does not show that preference-conditioned hypernetworks are ineffective in general.

It shows that the minimal function-preserving low-rank action-head hypernetwork tested here learned a multidimensional *local variation* in policy-parameter space, but that variation remained superimposed on a strongly shared generated displacement.

Therefore the architecture did not establish the distinct preference-indexed policy realizations required before testing the winner-rotation hypothesis semantically.

Primary artifacts:
- `talon_rl/v2h_actor_critic.py`
- `scripts/rl/v2h1_parameter_manifold_screen.py`
- `runs/v2h1_parameter_manifold-2026-09-24/v2h1_report.json`
- `runs/v2h1_parameter_manifold-2026-09-24/model_75.pt`
- `docs/contracts/preference_architectures/v2h-contract.md`
