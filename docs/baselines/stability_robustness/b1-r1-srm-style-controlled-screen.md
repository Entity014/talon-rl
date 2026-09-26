# B1-R1 — feed-forward SRM-style reconstruction

B1-R1 inherits B1-P2 and adds one auxiliary reconstruction head to the shared actor trunk. The head predicts rollout-only privileged targets `(tilt, height, contact)` from the existing 51-dimensional actor observation. It is not concatenated into the actor input, reward, deterministic monitor input, or normalization state. The 64 frozen monitor states remain evaluation-only.

For rollout observations `o`, target `y`, and shared trunk `h(o)`, the actor objective is

`L_actor = L_PPO + 0.10 * MSE(reconstruction_head(h(o)), y)`.

The reconstruction gradient reaches the shared actor trunk and reconstruction head only; critic optimization and B1-P2 adaptive actor-LR semantics remain unchanged. The target plumbing is time-major rollout storage and does not use evaluation trajectories. This is a controlled screen first; update-500 deterministic monitor gates remain the only training verdict.
