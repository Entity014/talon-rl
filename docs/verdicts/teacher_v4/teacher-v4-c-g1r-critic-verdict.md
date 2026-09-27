# Teacher V4 — V4-C G1-R Critic Verdict

Status: **FROZEN — G1-R FAIL: critic valid on 4/66 sets under the MC256 target; genuine, objective-dependent critic weakness (T valid, O partial, A and S invalid)**
Date: 2026-09-27
Contract: [teacher-v4-c-g1r-critic-contract.md](../../contracts/teacher_v4/teacher-v4-c-g1r-critic-contract.md) (frozen at `0cb3382`, before running)

The original result is unchanged: **V4-C G1 = FAIL under the preregistered
H32 critic contract** ([verdict](teacher-v4-c-g1-verdict.md)). G1-R is a
follow-up under a critic-independent long-horizon target.

## Run

`g1r_critic_evaluate.py` (commit `23c4308`) on the six `model_300.pt`
checkpoints; output `g1r_critic_evaluation.json` per run. Target: fixed
256-step Monte Carlo return from every scored state t = 0…63, γ = 0.99, no
bootstrap, stopping at the first true termination.

Determinism gate: the first 64 steps reproduce the stored G1 truncated EV on
all six runs. **Valid.**

## Result

| | critic valid (MC256) |
|---|---:|
| all sets | **4/66** |
| sets containing S | 0/42 |
| sets without S | 4/24 |
| m = 2 / 3 / 4 | 4/36 / 0/24 / 0/6 |
| seen / held-out | 1/36 / 3/30 |

Valid sets: {T,O} ×3 (two held-out, one seen) and {T,A} ×1 (held-out).

Fold-seed pass (critic criterion substituted, all other G1 criteria as
stored): **0/6**. The m = 4 anchor fails in 6/6. G1-R is not supported.

Per objective, over 42 set × run entries (EV pooled over all scored samples):

| objective | pooled EV median (min) | bias | corr | target var | prediction var |
|---|---|---:|---:|---:|---:|
| T | **0.934** (0.720) | 0.051 | 0.976 | 2.8e-1 | 2.0e-1 |
| A | −0.258 (−10.25) | −0.099 | 0.546 | 2.5e-3 | 5.3e-3 |
| O | 0.355 (−0.33) | −0.023 | 0.612 | 8.9e-3 | 3.3e-3 |
| S | −8.345 (−199.6) | −0.093 | 0.713 | 2.6e-4 | 3.6e-3 |

Termination within the 256-step window reaches 0.86 on a few sets
(e.g. G1-3 s73103 sets with A or S); most sets have none.

## Reading (per the interpretation fixed in the contract)

Outcome: **"MC256 critic still fails broadly → genuine critic validity /
generalization problem"**, and it is objective-dependent:

- T: valid everywhere (pooled EV ≥ 0.72, correlation 0.98).
- O: partially valid.
- A and S: invalid. For both, the critic's predictions vary more than the
  true long-horizon returns do (prediction variance 2× target variance for A,
  14× for S), so it responds to state or preference differences that do not
  change A/S returns over 256 steps.

This overturns the earlier post-hoc reading. The bootstrapped-target
diagnostic C1-D0 gave 56/66, and that was inflated by self-reference, as its
caveat warned. The original 0/66 was partly a target mismatch: the truncated
H32 target is also wrong for this critic, as T shows, since T is valid under
MC256 and invalid under H32. But the mismatch does not explain away the A and
S failures.

## Consequences

1. Critic validity for A and S is a real V4-C weakness, not an evaluation
   artifact. The same objective, A, is also the failing semantic endpoint in
   4/6 m = 4 anchors, so the critic and semantic failures may share a cause.
2. S's critic failing while its endpoint semantics mostly pass means correct
   S behavior does not currently depend on a valid S value estimate.
3. Held-out cardinality generalization remains unestablished.
