# M0.2 — Preference-Conditioned MOPPO Design Draft

Status: **M0.2a FROZEN — M0.2b TRAINING UNAUTHORIZED**

M0.1 established a stable scalarized locomotion substrate and proved that the
stock scalar reward can be represented as a five-component vector without
changing the scalar training signal. M0.2 re-enters the thesis contribution:
preference-conditioned multi-objective control. The first question is
locomotion preservation; preference trade-offs are evaluated only after that
gate passes.

## Frozen inheritance

M0.2 inherits the L0-A/M0.1 environment, action, reset, termination, command,
normalization, PPO rollout, checkpoint, and deterministic evaluation contracts.
It must not import B0/B1 P1/P2/S1/R1 state, scheduled-fixed-std state, or any
old preference checkpoint. The five objective order is:

`[progress, efficiency, contact, balance, limits]`.

The M0.1-derived reference case uses the **reference-scaled** uniform preference
`w_ref = [0.2, 0.2, 0.2, 0.2, 0.2]` together with an explicit `K=5`
scale in the actor loss. Thus `K * sum_k(w_k L_k)` equals `sum_k L_k`
for the uniform case. An implementation may store `[1,1,1,1,1]` as an
unnormalized reference coefficient, but it must produce the same loss and
gradient **when the supplied objective advantages satisfy
`sum_k A_k = A_scalar`**. This is conditional algebraic equivalence, not an
assumption that independently trained vector critics produce the same
advantages as M0.1. Reward reconstruction and actor-loss/gradient equivalence
are separate tests.

## Proposed model contract

- The preference vector `w` is normalized on the simplex and concatenated to
  the actor observation only at the policy trunk input. It is not a physical
  environment observation or reward input. The deterministic monitor receives
  an explicit predeclared `w_eval`; it does not infer `w` from monitor state.
- The critic is vector-valued and predicts one value per objective. It receives
  the training observation and `w`, but its outputs remain objective-separated.
- The actor uses deterministic `tanh(actor_mean(obs, w))` at evaluation.
- The five reward components are kept unnormalized for the reference-equivalence
  check. If objective normalization is introduced later, it requires a new
  design gate and manifest; it is not part of this draft.

## Optimization semantics

The first implementation uses **per-objective late weighting**. For each
objective `k`, use the same joint-policy ratio `r_t` and compute an independent
clipped PPO surrogate `L_k = min(r_t A_k, clip(r_t,1-e,1+e) A_k)`. The actor
loss is `K * sum_k(w_k L_k)`; it is not `sum_k(w_k A_k)` followed by one clip.
Preference never enters environment reward computation. The conditional
algebraic case must match the M0.1 actor loss and gradient within a declared
numerical tolerance; this is covered by a dedicated unit test. Production
M0.2a uses a new first-layer input shape and independently trained vector
critic, so actual training gradients are not expected to match M0.1 after
critic learning begins.

The numerical contract is fixed before smoke: actor-loss absolute and relative
error `<= 1e-6`, and actor-gradient maximum absolute and relative L2 error
`<= 1e-6`. These deterministic criteria must not be relaxed after observing a
run.

The vector critic loss is explicitly `mean_k MSE(V_k, R_k)` with value
coefficient `1.0`. This is a declared M0.2 vector-critic scale, not a claim
that the critic loss is numerically identical to M0.1's scalar critic loss.
Its finite-value, gradient, and locomotion-preservation behavior are tested in
M0.2a before preference diversity is introduced.

The rollout stores objective rewards, objective returns/advantages, `w`, and
the old joint log probability. No preference-conditioned state may be inferred
from checkpoint history. Learned standard deviation, adaptive-KL behavior (if
retained from the reference recipe), and optimizer state are checkpointed
explicitly.

## Two-phase preference protocol

M0.2 is split into two predeclared experiments:

1. **M0.2a — fixed-preference MOPPO preservation:** actor preference input,
   vector critic, and per-objective late weighting are enabled, but `w_ref` is
   fixed for the complete three-seed run. This asks whether the changed
   preference-conditioned/vector-critic MOPPO structure preserves locomotion;
   it is not a byte-equivalence test. No trade-off claim is made.
2. **M0.2b — preference-conditioned training:** only after M0.2a passes,
   sample `w` from the frozen preference distribution during training. Uniform
   preference remains a deterministic evaluation condition, while fixed
   non-uniform preferences are evaluated for trade-offs. A checkpoint trained
   only with `w_ref` is not used as evidence of preference generalization.

The preference distribution, non-uniform evaluation set, and sampling seed are
configuration fields in the M0.2b manifest, not runtime choices. No preference
sweep or adaptive preference curriculum is authorized by this draft.

## Instrumentation and acceptance

Every update logs preference, objective reward/return/advantage statistics,
vector value loss, policy loss, entropy, KL/clip metrics, action statistics,
and deterministic monitor results. The monitor remains evaluation-only and
must prove model/RNG/optimizer/normalizer non-mutation.

The preservation gate requires, for every seed, finite artifacts, complete
checkpoints, no preference-state leakage, and the unchanged Gate-0/L0
locomotion safety and tracking thresholds. Trade-off metrics are not used to
rescue a preservation failure.

## Design-freeze gate (before training authorization)

The following must be specified and hashed before any M0.2 run:

- actor preference injection and critic output shapes;
- `K`-scaled uniform actor-loss equivalence, including loss and gradient tests;
- algebraic test with constructed `A_k` satisfying `sum_k A_k = A_scalar`,
  plus a separate production vector-critic/GAE preservation test;
- per-objective clipped-surrogate equations and log-probability semantics;
- vector-critic `mean_k MSE` scale and coefficient;
- separate M0.2a and M0.2b manifests and authorization gates;
- preference normalization/sampling and fixed evaluation set;
- objective normalization decision;
- optimizer, standard-deviation, rollout, and checkpoint semantics;
- preservation and trade-off acceptance criteria;
- lifecycle/artifact schema and inherited M0.1 hashes;
- unit tests, production smoke, and a manifest with `training_authorized=false`.

M0.2a and M0.2b have separate manifests. M0.2a is the only manifest eligible
for freeze after the algebraic and production smoke gates. M0.2b remains a
design placeholder until M0.2a passes; its preference distribution and
non-uniform evaluation set are intentionally not frozen here.

Three-seed checkpoints are an experiment-completion gate, not a design-freeze
requirement. This document therefore does not authorize training yet.

## Explicit exclusions

No RMA, exteroception, terrain randomization, command-range expansion, ACAPS,
SRM, recurrent policy, adaptive preference curriculum, reward coefficient sweep,
or retrospective use of old B0/B1/MOPPO checkpoints is included in M0.2.
