# AI-C0/C1 Critic-Only Representation Repair Contract

Status: **PREDECLARED — FROZEN ACTOR, NO ACTOR RL**
Date: 2026-09-25

## Treatment isolation
Freeze the AI-H1 u75 actor exactly. Change only critic representation.

Control critic:
    52 -> 128 -> 128 -> 128 -> 4

Treatment critic:
    52 -> 256 -> 256 -> 128 -> 4

Both retain:
- shared critic body;
- four objective-specific scalar outputs;
- input = observation + preference;
- Foundation V2 objective definitions;
- gamma=.99;
- no semantic labels.

## AI-C0 actor invariance
Treatment model must reproduce the AI-H1 u75 actor exactly:
- deterministic action max error <=1e-6;
- stochastic action/log-prob max error <=1e-6 under matched RNG;
- simplex-tangent action-Jacobian error <=1e-6;
- authority metrics unchanged <=1e-6 relative;
- checkpoint roundtrip <=1e-6.

## AI-C1 fixed-policy capacity test
Actor remains frozen.

Train critic only on current-policy support from 6 independent reset seeds × T/A/O/S/C × 64 steps.
Hold out 2 additional reset seeds × all five preferences.
Semantic evaluation suites 840001..840004 are never used for fitting.

Target:
- phase-local H32 objective returns for critic fitting.
- MC64 is evaluation-only.

Optimization:
- Adam, lr=3e-4;
- batch=2048;
- 5000 critic-only steps;
- MSE over four objective values;
- no actor gradient.

Evaluation:
- held-out dense resets;
- semantic suites;
- H32 and MC64 EV;
- negative-EV fraction;
- bias;
- by objective head and preference;
- survival.

## Frozen AI-C1 PASS
Treatment must satisfy on BOTH held-out dense and semantic suites:
- H32 EV mean >0;
- MC64 EV mean >0;
- H32 negative fraction <=.25;
- MC64 negative fraction <=.25;
- min survival >=.95.

Additionally on semantic suites:
- Angular H32 EV >0;
- Smoothness H32 EV >0.

And treatment must improve u75 old-body semantic late/H32 failure without actor change.

PASS -> authorize AI-C2 paired RL rerun.
FAIL -> minimal critic capacity escalation rejected; no actor or semantic H2 change.
