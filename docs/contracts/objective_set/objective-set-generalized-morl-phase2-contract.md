# Objective-Set Generalized MORL — Phase 2 Contract

Status: **PREDECLARED — DESIGN/FREEZE BEFORE IMPLEMENTATION**
Date: 2026-09-25

## 1. Scientific question

Can the validated Phase-1 controller principles be lifted from a fixed-index four-objective interface to a single objective-set-conditioned MORL formulation that supports variable active-set cardinality and unseen combinations of a predefined objective vocabulary without architecture redesign or objective-specific patches?

Phase 2 is **not** a Smoothness-repair branch.

Phase 1 remains frozen evidence:
- fixed vocabulary/order: T/A/O/S;
- durable preference authority established;
- T/A/O endpoint semantics valid at the validated u30 operating point;
- S semantics remain a documented heterogeneous closed-loop limitation;
- H2b semantic durability was not authorized because H2a was not 4/4 valid.

Canonical Phase-1 semantic checkpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

## 2. Scope and claim boundary

Primary Phase-2 scope is G0 -> G1 -> G2.
G3 unseen-objective identity transfer is a stretch goal and is not authorized unless G2 passes.
Authorized G2 claim:

> A single objective-set-conditioned MORL controller supports variable-size subsets and previously unseen combinations of a predefined objective vocabulary without architecture redesign.

Phase 2 must not claim arbitrary or unseen-objective semantic generalization.

## 3. Objective-set abstraction

For active set size m:

    O = {(e_i, w_i)} for i=1..m

where e_i represents objective identity, w_i >= 0, and weights sum to one over the active set.

Vocabulary:

    T = velocity_tracking
    A = angular_stability
    O = orientation_stability
    S = control_smoothness

Inactive objectives are absent from the set. Objective order in memory is non-semantic.

## 4. Objective token contract

G0-G2 use known-vocabulary identity tokens, not semantic-language/reward encoders.
Token dimension d_e is configurable and independent of active-set cardinality.
Authorized G2 claim:

> A single objective-set-conditioned MORL controller supports variable-size subsets and previously unseen combinations of a predefined objective vocabulary without architecture redesign.

Phase 2 must not claim arbitrary or unseen-objective semantic generalization.

## 3. Objective-set abstraction

For active set size m:

    O = {(e_i, w_i)} for i=1..m

where e_i represents objective identity, w_i >= 0, and weights sum to one over the active set.

Vocabulary:

    T = velocity_tracking
    A = angular_stability
    O = orientation_stability
    S = control_smoothness

Inactive objectives are absent from the set. Objective order in memory is non-semantic.

## 4. Objective token contract

G0-G2 use known-vocabulary identity tokens, not semantic-language/reward encoders.
Token dimension d_e is configurable and independent of active-set cardinality.

For G0 compatibility initialization:
- T/A/O/S receive deterministic distinct basis-compatible tokens;
- token construction must permit exact recovery of legacy 4D preference coordinates;
- learned residual token features may be added only through a zero-initialized path so G0 can begin function-equivalent to Phase 1.

Adding/removing an active token must not require changing network layer shapes.
A new unseen objective identity belongs to G3.

## 5. Set aggregation rule

The objective context must be permutation invariant:

    z_O = A({(e_i,w_i)})

Permitted compatibility form:

    z_base = sum_i w_i e_i
    z_res  = rho(sum_i phi([e_i,w_i]))
    z_O    = concat_or_project(z_base, z_res)

The residual path is zero-initialized at G0.

Forbidden:
- concatenating tokens in T/A/O/S order;
- objective-index-specific branches;
- separate aggregators by cardinality;
- positional embeddings tied to input order.
## 6. Hard permutation-invariance gate

For identical state and mathematically identical objective set, evaluate all token permutations.

Required:

    max_abs(action_perm - action_ref) <= 1e-6
    max_abs(pre_tanh_perm - pre_tanh_ref) <= 1e-6
    max_abs(value_perm - value_ref) <= 1e-6

This is a hard gate. Failure blocks G1.

## 7. Actor interface

Legacy:

    a = pi(s, w_4D)

Phase 2:

    a = pi_set(s, O) = pi_state_family(s, z_O)

Preference information enters through shared set context only.
No objective-specific actor routing is allowed.
The authority-isolated principle is retained: state trunk is state-only and set context owns the preference-dependent policy-family path.

## 8. Cardinality-independent critic interface

The fixed four-head critic is removed as the public Phase-2 value interface.

For every active objective token:

    V_i = V(s, e_i, z_O)

The same critic parameters are queried for every token and return one scalar per query.
No objective-specific heads or routing are allowed.

For G0 compatibility, querying T/A/O/S must reproduce the corresponding legacy four-head values at the frozen checkpoint.

## 9. G0 — representation equivalence

Purpose: prove that replacing the fixed-index API does not itself change the validated controller.
Use exact Phase-1 T/A/O/S vocabulary and exact same preference vectors.

### G0-A structural parity

On a frozen state bank from validated Phase-1 suites:
- deterministic pre-tanh parity <= 1e-6 max abs;
- deterministic action parity <= 1e-6 max abs;
- stochastic log-prob parity <= 1e-6 max abs for identical stored pre-tanh samples;
- queried T/A/O/S critic-value parity <= 1e-6 max abs;
- permutation hard gate PASS.

No training is allowed in G0-A.
### G0-B behavioral parity

Run frozen H2a matched-reset protocol with compatibility-initialized set model.
Require:
- T endpoint PASS preserved;
- A endpoint PASS preserved;
- O endpoint PASS preserved;
- endpoint survival >= .95;
- critic validity remains within frozen H2a gate;
- continuum monotonicity >= .65;
- continuum endpoint-between >= .65;
- center compromise >= .75.

S is reported exactly but is not required to become valid.
Any apparent S improvement is descriptive only.
G0 failure blocks G1.

## 10. G1 — variable known-objective cardinality

Purpose: test whether one unchanged network operates across active-set sizes.

Primary cardinalities:

    m = 2, 3, 4

m=1 specialists are controls only.

For active set size m:
- center weights are uniform, 1/m;
- heavy endpoint i uses 0.70 on i;
- remaining 0.30 is divided uniformly among other active objectives.

Thus m=4 reproduces [.7,.1,.1,.1].

### G1 training distribution

The same actor/critic instance and parameterization is used for every cardinality.
Training samples active sets and preferences without cardinality-specific modules.

### G1 cardinality holdout folds

Fold G1-2:
- train with m=3 and m=4 sets;
- evaluate all m=2 sets zero-shot.

Fold G1-3:
- train with m=2 and m=4 sets;
- evaluate all m=3 sets zero-shot.

No architecture, optimizer rule, loss definition, token dimensionality, or evaluation rule may change between folds/cardinalities.
## 11. G2 — unseen objective combinations

Purpose: separate compositional set generalization from unseen objective identity.
Every T/A/O/S identity must be present during training.
Held-out combinations never appear as exact active sets during training.

### Pair holdout folds

    P1: hold out {T,A} and {O,S}
    P2: hold out {T,O} and {A,S}
    P3: hold out {T,S} and {A,O}

Remaining pairs plus permitted non-held-out cardinalities remain available.

### Triple holdout folds

    R1: {T,A,O}
    R2: {T,A,S}
    R3: {T,O,S}
    R4: {A,O,S}

Other triples, permitted pairs, and the full four-objective set remain available.
The exact held-out set must not enter replay, curriculum, validation-driven training, or checkpoint selection.
Evaluation is zero-shot with respect to combination identity.

## 12. Standard semantic evaluation for any active set

### Endpoint semantics
For every active objective i, compare i-heavy against matched center.
Require:
- normalized objective direction correctness >= .75;
- physical semantic-proxy direction correctness >= .75;
- endpoint survival >= .95.

Frozen physical proxies:
- T: tracking error, lower is better;
- A: xy angular velocity, lower is better;
- O: body tilt, lower is better;
- S: action rate, lower is better.

### Center/compromise
Center lies inside active-heavy envelope in >= .75 of objective x suite cells.

### Pairwise edges
Evaluate every heavy-heavy edge at alpha = 0,.25,.5,.75,1.
Aggregate monotonicity >= .65.
Aggregate endpoint-between >= .65.

### Sampled interior
Use a frozen interior Dirichlet sample set per active set.
Report preference-action separation, objective response, survival, and heavy-envelope containment.
Interior samples are characterization only and may not tune training.
## 13. Authority, durability, critic, and robustness gates

For every G1/G2 evaluation family:

### Preference authority
Changing set weights at fixed matched states must produce nontrivial action response through the designated set-conditioning path.
Masking/removing set context must reduce the validated preference-response metric according to the Phase-1 causal-authority principle.

### Authority durability
Evaluate authority prospectively across training checkpoints.
Checkpoint selection may use engineering eligibility but must not use semantic endpoint scores.

### Critic validity
The shared token-query critic must pass fresh held-out value validation for every active token.
No private objective critic head may be introduced as repair.

### Robustness
Report survival, held-out reset robustness, PPO ratio/action-log-prob consistency, finite outputs, and collateral tracking behavior.
Engineering failure blocks semantic interpretation for the affected cell.

## 14. Phase-2 semantic success rule

Phase 2 does not redefine the frozen Phase-1 S limitation.

G1/G2 method-level success requires:
1. structural permutation gate PASS;
2. variable-cardinality / held-out-combination execution without architecture change;
3. preference authority and critic/robustness gates PASS;
4. Phase-1-valid T/A/O retain endpoint semantic validity whenever present;
5. S is evaluated identically but S PASS is not required for the G1/G2 generalization claim.

If S becomes valid without objective-specific intervention, report it as emergent and require independent confirmation before expanding the claim.
If T/A/O systematically degrade after generalization, Phase 2 FAILS even if the interface works.

## 15. Forbidden adaptations

During G0-G2 it is forbidden to:
- add objective-specific actor branches;
- add objective-specific critic heads;
- add S-specific or other objective-specific auxiliary losses;
- redefine an objective reward to rescue a failing semantic gate;
- change architecture with active-set cardinality;
- change token dimension between folds;
- introduce order-dependent positional encoding;
- tune split definitions after results;
- select checkpoints using held-out semantic outcomes;
- use G2 held-out combinations in training or repair.

## 16. Stop rules

G0:
- permutation or numerical parity failure -> implementation repair only; G1 blocked.
- behavioral semantic regression after numerical parity -> investigate protocol validity; do not train around it.

G1:
- inability to operate on held-out cardinality -> cardinality generalization FAIL.
- broad authority/critic/robustness failure -> stop before G2 and localize engineering substrate.
- T/A/O semantic failure across variable cardinalities -> stop before G2; no objective-specific patch.

G2:
- systematic loss of authority or T/A/O semantic validity on held-out combinations -> compositional generalization not established.
- no post-hoc combination-specific repair is allowed inside the same G2 claim.

G3 remains unauthorized unless G2 passes structural, engineering, and semantic gates.

## 17. G3 stretch boundary

G3 changes the question from known-vocabulary composition to unseen-objective representation.
A G3 token must contain semantic information derived from an objective specification rather than learned ID lookup alone.
Any G3 representation requires a separate preregistered contract.
No G3 implementation work is authorized here.

## 18. Phase relationship

Phase 1 is frozen empirical foundation, not overwritten by Phase 2.

Phase 2 tests whether the same principles survive a generalized objective interface:

    explicit objective identity
      -> permutation-invariant set context
      -> preference authority
      -> shared objective-conditioned value estimation
      -> standardized per-objective semantic validation

Core distinction:

    preference authority != semantic correctness

Generalizing the interface must preserve this distinction rather than assuming one implies the other.
