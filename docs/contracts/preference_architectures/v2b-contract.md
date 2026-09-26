# V2-B Contract — Single-Site FiLM Escalation

Status: PREDECLARED BEFORE V2-B TRAINING

## Frozen
- Foundation V2 in full
- RV1 direct preference path
- V2-A 16D learned preference embedding path
- seed 73001
- 75-update budget
- expanded current-policy critic support
- critic body / linear heads / ridge lambda = 1
- repaired PPO/action-logprob semantics
- objectives/rewards
- semantic evaluator, matched reset suites, and thresholds

## Treatment only
One FiLM modulation site on the final shared actor hidden feature h, before concatenating the V2-A embedding:

    [gamma(w), beta(w)] = Linear(w)
    h_film = (1 + gamma(w)) * h + beta(w)
    actor_features = concat(h_film, embedding(w))

The FiLM generator weight and bias are zero-initialized, therefore gamma=0 and beta=0 exactly at V2-B0 initialization.

## Interpretation
This is B-as-escalation:
RV1 direct conditioning + V2-A embedding + one additional FiLM site.

Any gain over V2-A is interpreted as incremental gain from one-site modulation on top of the embedding architecture, not FiLM-alone gain.

## Ladder
V2-B0 identity/function-preserving gate
V2-B1 FiLM authority/sensitivity screen
V2-B2 exact Foundation-V2 semantic gate
V2-B3 multi-seed/full validation only if V2-B2 passes

## Stop rule
If single-site FiLM passes the frozen semantic contract, stop architecture escalation. No multi-site FiLM, adapters, routing, or auxiliary semantic losses are authorized before that result.
