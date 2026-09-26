# AI-C2 Paired Actor-Updating Critic Compatibility Contract

Status: PREDECLARED
Date: 2026-09-25

## Question

Does the wide critic capacity repair remain compatible with continued learning of the robust high-authority actor, relative to the original critic representation, while all actor-side and training semantics remain fixed?

## Starting policy

Authoritative robust checkpoint:

    runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt

Actor/log_std are copied exactly into both arms.

## Sole treatment

CONTROL:
    critic body = 52 -> 128 -> 128 -> 128 -> 4

TREATMENT:
    critic body = 52 -> 256 -> 256 -> 128 -> 4

Actor architecture is identical.

## Critic equilibration before actor learning

Before the first actor update:
- freeze the robust control-u20 actor;
- collect the same reset/preference support dataset for both critics;
- fit each critic body+head on the same phase-local H32 targets;
- use identical optimizer, batch schedule, fitting steps, and random minibatch indices;
- do not update actor parameters.

After this equilibration, actor parameters must still match exactly across arms.

## Coupled continuation

Both arms use:
- identical actor initialization;
- identical actor optimizer state from robust control-u20;
- GAE lambda = .95;
- gamma = .99;
- continuous one-model preference schedule;
- projected PPO/tail-conflict repair;
- active tail descent kappa = .05;
- original UnitreeA1FlatEnvCfg reset/training distribution;
- same anchor/reset/support schedule;
- same adaptive support budget and freshness rule;
- same actor learning rate and clipping;
- same update count and random seed schedule;
- critic body frozen during actor learning;
- critic linear head refreshed from current-policy representative support identically in both arms.

Initial screen:

    robust-u20 -> paired continuation for 10 actor updates

## Primary gates

At endpoint and fresh measurement-only audit:

Authority:
- pairwise preference-action separation retention >= .90 relative to AI-C2 start;
- simplex-tangent action-Jacobian retention >= .90.

Critic:
- H32 EV mean > 0;
- MC64 EV mean > 0;
- H32 negative fraction <= .25;
- MC64 negative fraction <= .25;
- fresh minimum survival >= .95.

Robustness:
- frozen semantic-suite minimum survival = 1.00;
- held-out reset minimum survival >= .95;
- no new systematic termination topology.

PPO:
- post-refresh ratio invariant <= 1e-4.

## Causal interpretation

AI-C2 PASS for the wide critic requires:
- treatment satisfies every primary/no-regression gate;
- preference authority remains intact;
- critic remains valid under actor learning;
- robustness remains intact.

The narrow control is diagnostic:
- if narrow critic regresses while wide remains valid, this supports critic-capacity compatibility as the causal repair;
- if both remain valid, wide capacity is compatible but a strict necessity claim is not supported in this robust-u20 regime;
- if wide fails, AI-H2 remains blocked.

## Authorization

PASS of the wide treatment authorizes AI-H2 semantic forgetting evaluation.

AI-C2 does not itself test semantic forgetting or winner rotation.
