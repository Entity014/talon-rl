# Phase-1 D0 — Canonical Controller Freeze Manifest

Status: **FROZEN**
Date: 2026-09-25

## Canonical checkpoint

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

SHA256:

    30b34d29a003f44b385269a61e1fb0a2000702d275dd09701f53d23d3c726e56

## Canonical implementation hashes

    scripts/rl/authority_isolated_simplex_edge_retention_train.py
    a23498c37ac7d398b0a35eaea47a4f3953e265e2a0f55f362883d7f469de410b

    talon_rl/authority_isolated_wide_critic.py
    502eafda3ffbdd4d4c5873a071f6c37408f7e03c6cd03662e4d46d942be85312

    talon_rl/t4_actor_critic.py
    e892baa39d2ea11ad1bafb963a0521b950fe6583262862cfd2721cb3b50218a2

    talon_rl/t3b_objectives.py
    8876b25dcbd9ef13b38b87dbc1131f6623a9fe22cbfa74c60b86687c9cce2952

## Canonical evidence hashes

    docs/verdicts/authority/authority-isolated-h2a-u30-semantic-validity-verdict.md
    5c603cade197fbfc29bbe3792a2126ec42aa6534f53932a4f94c78098d1de98d

    docs/verdicts/authority/authority-isolated-s-trajectory-temporal-decomposition-verdict.md
    af425fb6d1c006b30ff57a99dab86b50800cefc20c2001b9e00d3e7c4b6d0d8e

## Runtime snapshot

    Python      3.11.15
    PyTorch     2.7.0+cu128
    CUDA        12.8
    Platform    Linux-7.0.0-31-generic-x86_64-with-glibc2.39
    Git commit  cfe9376d4012e56967babd5df32620e824560c16

## Frozen controller contract

Preference order:

    [T, A, O, S]

Endpoint preferences:

    T = [.7,.1,.1,.1]
    A = [.1,.7,.1,.1]
    O = [.1,.1,.7,.1]
    S = [.1,.1,.1,.7]
    C = [.25,.25,.25,.25]

Validated semantic interpretation:

    T = velocity tracking
    A = angular stability
    O = orientation stability
    S = control smoothness

The final Phase-1 method is frozen after D0.

No further architecture, reward, scalarization, authority-retention, critic, or PPO modification is allowed on the deployment-critical Phase-1 branch.

Any later research modification must use a new branch/name and cannot silently supersede this checkpoint.

## Authorized scientific claim

> The final Phase-1 method substantially improves preference authority, authority durability, robustness, and simultaneous semantic validity across three of four objectives, while Smoothness remains an unresolved heterogeneous closed-loop semantic limitation.

## Not claimed

- 4/4 semantic validity
- H2b semantic durability
- semantic forgetting solved
- objective generalization
- V3 variable-cardinality learning/generalization

## Next stage

    D1 — Phase-1 Multi-Seed Final Characterization
