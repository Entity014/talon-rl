# `experiments`

One directory per experiment line. Each has a `README.md` listing its files with
a one-line description taken from the file's own docstring, so the description
lives with the code rather than beside it.

Regenerate them all after adding or describing a script:

```
python scripts/rl/experiments/generate_readmes.py --apply
```

`shared/` is different from the rest: it holds the modules other scripts import,
including the base classes the converted audits build on. Nothing there is a
one-shot experiment.

624 scripts, 202 described, 422 still without a docstring.

| directory | scripts | described |
|---|---:|---:|
| [`post_v2`](post_v2/README.md) | 177 | 51 |
| [`authority_isolated`](authority_isolated/README.md) | 89 | 12 |
| [`v2b`](v2b/README.md) | 44 | 0 |
| [`phase3`](phase3/README.md) | 30 | 3 |
| [`shared`](shared/README.md) | 30 | 13 |
| [`m0`](m0/README.md) | 29 | 23 |
| [`phase1`](phase1/README.md) | 26 | 1 |
| [`b1`](b1/README.md) | 25 | 10 |
| [`rv1`](rv1/README.md) | 20 | 0 |
| [`b0`](b0/README.md) | 15 | 11 |
| [`l0`](l0/README.md) | 15 | 14 |
| [`phase5`](phase5/README.md) | 12 | 0 |
| [`v1a`](v1a/README.md) | 12 | 11 |
| [`objective_set`](objective_set/README.md) | 11 | 0 |
| [`v1b`](v1b/README.md) | 11 | 10 |
| [`r1`](r1/README.md) | 10 | 10 |
| [`v1`](v1/README.md) | 9 | 9 |
| [`phase4`](phase4/README.md) | 7 | 0 |
| [`v1c`](v1c/README.md) | 6 | 6 |
| [`misc`](misc/README.md) | 5 | 3 |
| [`update`](update/README.md) | 4 | 2 |
| [`final`](final/README.md) | 3 | 3 |
| [`pivot_p0`](pivot_p0/README.md) | 3 | 3 |
| [`prospective`](prospective/README.md) | 3 | 0 |
| [`separated_anchor`](separated_anchor/README.md) | 3 | 0 |
| [`v2a`](v2a/README.md) | 3 | 0 |
| [`v2k`](v2k/README.md) | 3 | 0 |
| [`p1`](p1/README.md) | 2 | 2 |
| [`pivot_p1`](pivot_p1/README.md) | 2 | 2 |
| [`preference`](preference/README.md) | 2 | 0 |
| [`relational`](relational/README.md) | 2 | 1 |
| [`semantic_gate`](semantic_gate/README.md) | 2 | 1 |
| [`v2c`](v2c/README.md) | 2 | 0 |
| [`v2h`](v2h/README.md) | 2 | 0 |
| [`v2pf`](v2pf/README.md) | 2 | 0 |
| [`foundation`](foundation/README.md) | 1 | 0 |
| [`g1`](g1/README.md) | 1 | 1 |
| [`trajectory`](trajectory/README.md) | 1 | 0 |
