# V2-B Fixed-Stream Multi-Update Audit

Status: INVALIDATED FOR CAUSAL MULTI-UPDATE INFERENCE

Date: 2026-09-24

The fixed-stream diagnostic reused batches collected from the frozen base policy while repeatedly updating cloned actors.

After the first update, PPO likelihood ratios relative to the stored old log-prob became extremely off-policy:
- step 1: ratio max error already ~8.6-11
- step 2: ~47-107
- later steps remained far outside the on-policy training regime.

Therefore PPO clipping dominated the repeated-update path. The large step-16 redistribution (e.g. lambda=1 Tracking share increase and Angular/Smoothness share loss) cannot be interpreted as the mechanism operating in the real training loop, where a fresh current-policy batch is collected each update and the ratio invariant begins near one.

Retained finding from the prior valid static audit:
- at a single on-policy batch, lambda=1 does not statically suppress Angular and heavy-objective alignment remains strong.

Rejected use:
- do not use the repeated fixed-stream path as evidence for parameter-only optimization drift.

Replacement diagnostic:
- paired checkpoint-path audit on actual lambda=.95 and lambda=1.0 training checkpoints (u10/u25/u50/u75)
- matched fresh on-policy rollouts at each checkpoint
- cross-compute both lambda values on each checkpoint/batch
- separate estimator effect within checkpoint from learned-policy/path divergence across checkpoints.