# `v1`

9 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `post_v1_d1_aggregate.py` | 36 | — | Aggregate the frozen D1 fixed-specialist endpoint audit. |
| `post_v1_d1_specialists.py` | 72 | isaac trains | Post-V1 D1: train three fixed-preference specialists, then evaluate terminals. |
| `post_v1_d2_conditioning_audit.py` | 76 | isaac | Post-V1 D2: read-only layer-wise preference-conditioning audit. |
| `post_v1_d3_screen.py` | 70 | isaac trains | Post-V1 D3 one-seed, 100-update FiLM diagnostic screen. |
| `post_v1_d4a_audit.py` | 80 | isaac trains | Post-V1 D4-A: read-only preference recoverability and gradient audit. |
| `post_v1_d4b_screen.py` | 64 | isaac trains | Post-V1 D4-B one-seed 100-update auxiliary-loss screen. |
| `post_v1_d4c_alignment_audit.py` | 129 | class | Do D4-B's preference-dependent actions point where the D1 specialists do? |
| `post_v1_d5a_specialist_compatibility_audit.py` | 199 | isaac | Post-V1 D5-A: read-only compatibility audit for fixed-preference specialists. |
| `post_v1_d5b_alignment_pilot.py` | 88 | isaac trains | Post-V1 D5-B: compatibility-aware specialist-alignment pilot. |

## Run directories these touch

- `runs/post_v1_d1-2026-09-22`
- `runs/post_v1_d3-2026-09-22`
- `runs/post_v1_d4b-2026-09-22`
- `runs/post_v1_d4c-2026-09-22`
- `runs/v1c_confirmatory_seed0-2026-09-22`
