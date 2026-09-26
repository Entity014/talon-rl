# One-Model Explicit Continuous Policy-Family Contract

Status: **PREDECLARED — FINAL FORMULATION BRANCH**

Date: 2026-09-25

## Non-negotiable product requirement

The final method must satisfy all four:

    TRAINING ARTIFACT   = 1 model
    DEPLOYMENT ARTIFACT = 1 model
    continuous w ∈ Δ^3 supported
    no per-objective specialist training / checkpoint selection

Any candidate violating one of these is excluded from the main roadmap.

## Scientific hypothesis

The validated V2-B conditional actor may be limited by forcing the full preference simplex to share one evolving actor parameter realization.

Test whether a single jointly-trained model that generates a substantial preference-indexed policy block can represent a more stable continuous policy family:

    w -> H_phi(w) -> theta_family(w)
    (s,w,theta_family(w)) -> action

This differs from V2-H, which generated only a low-rank delta on the final action head.

## Frozen foundation

- V2-B base checkpoint / Foundation V2
- repaired PPO transformed-action semantics
- GAE lambda=.95
- normalized 4D objectives
- critic/support/freshness contract unchanged
- training/evaluation preference suites unchanged
- semantic thresholds unchanged
- no replay, rehearsal, gradient surgery, temporal/context loss, or local-region intervention

## Architecture treatment

### Shared part

Retain V2-B exactly through:

    observation + direct w path
    actor trunk
    learned preference embedding
    single-site FiLM

This yields V2-B feature vector:

    x(s,w) ∈ R^144

### Preference-generated family block

Replace the direct shared action readout with a function-preserving conditional residual block:

    z = ELU((W1_0 + ΔW1(w)) x + b1_0 + Δb1(w))
    μ = μ_V2B(x) + Δμ_family(x,w)

where the family residual uses:

    Δμ_family(x,w)
      = (W2_0 + ΔW2(w)) z + b2_0 + Δb2(w)
        - family_base(x)

Implementation may equivalently express this as a zero-authority residual block, but initialization must satisfy exact V2-B functional equivalence.

Preference-conditioned parameter deltas are generated using learned full-rank bases:

    ΔW_l(w) = Σ_k c_k(w) B_{l,k}

with shared coefficient network H_phi and latent, semantically unlabeled bases.

Frozen dimensions:

    family hidden dim = 128
    number of bases K = 8
    coefficient hidden dim = 32

Generated tensors cover both:

    144 -> 128 family hidden layer
    128 -> 12 family action layer

plus biases.

No objective-labelled expert/module assignment is allowed.

## H0 — exact function preservation

Before training, initialized one-model family must be numerically equivalent to V2-B for:

- deterministic action mean
- stochastic sampled action under matched RNG
- pre-tanh log-prob
- value output
- environment-applied action
- PPO pre-update ratio
- checkpoint save/load roundtrip

Across frozen T/A/O/S/C preferences and multiple reset suites:

    max action error <= 1e-6
    max log-prob error <= 1e-6
    max value error <= 1e-6
    max PPO ratio error <= 1e-6

H1/H2 are blocked until H0 passes.

## H1 — policy-family authority / non-degeneracy

Short joint training only; no semantic verdict yet.

Must establish all:

1. coefficient map c(w) differs across T/A/O/S/C;
2. generated full-rank parameter deltas differ across preferences;
3. generated family manifold has >=2 meaningful directions;
4. family-block masking materially reduces fixed-state preference-conditioned action separation;
5. generated block contributes nonzero action authority;
6. common-component dominance is lower than in V2-H or absolute preference-specific generated parameter variation is materially larger;
7. Foundation V2 / PPO ratio / critic freshness / survival remain valid.

Primary H1 metrics:

    pairwise ||c(w_i)-c(w_j)||
    pairwise ||Δθ(w_i)-Δθ(w_j)||
    cosine matrix of Δθ(w)
    effective rank of generated policy realizations
    family-block action authority
    preference Jacobian with/without family block
    common-vs-specific generated-parameter energy

No threshold is tuned after observing H1.

## H2 — exact semantic accumulation gate

Only if H1 passes.

Use the exact frozen endpoint + continuum semantic contract.

Primary question:

> Does semantic competence accumulate across T/A/O/S while retaining a correct continuous preference-to-behavior map, rather than rotating among winners?

Report:
- endpoint PASS counts over checkpoints;
- continuum ordering / interpolation;
- continuous semantic margins;
- first PASS -> retained fraction;
- winner-rotation frequency;
- simultaneous competence count;
- center compromise behavior.

## H3 — capacity-matched control

Only if H2 meaningfully improves.

Compare against a dense preference-independent residual block with approximately matched parameter count and initialization/training budget.

Required inference:

    policy-family > dense control

before attributing gains to preference-indexed parameter generation rather than capacity.

## H4 — multi-seed + deployment

Only after H2/H3 pass:
- 3 seeds minimum;
- latency;
- FLOPs;
- model memory;
- checkpoint size;
- continuous-w inference stability.

## Stop rules

Stop immediately if:

- H0 fails exact equivalence;
- H1 collapses to common generated parameter displacement;
- H1 family block has negligible action authority;
- H2 reproduces winner rotation / semantic non-accumulation;
- dense capacity control explains the gain.

Do not tune basis count, generator depth, or temperature after an H1/H2 failure within this branch.
