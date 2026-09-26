# AI-H2 Semantic Accumulation / Forgetting Contract

Status: PREDECLARED
Date: 2026-09-25

## Authorization

AI-H2 is authorized by:

    docs/methods/general/authority-isolated-ai-c2-compatibility-addendum.md

Reference critic:

    narrow compatible critic

No new engineering repair is introduced in H2.

## Scientific question

When preference authority is both created and durably preserved, does semantic competence accumulate across T/A/O/S, or does winner rotation / semantic forgetting still occur?

This is the first semantic test after closing:
- actor authority ownership;
- authority durability;
- output/headroom robustness;
- critic compatibility.

## Starting checkpoint

Exact compatible narrow AI-C2 endpoint:

    runs/authority_isolated_ai_c2_edgec2_narrow-2026-09-25/model_10.pt

Exact continuation state:

    runs/authority_isolated_ai_c2_edgec2_narrow-2026-09-25/resume_state.pt

This corresponds to global continuation update u30.

## Frozen training stack

Keep unchanged:
- authority-isolated actor;
- narrow critic body 128-128-128;
- bounded all-simplex-edge retention;
- gamma edge floor = .90;
- rho = .25;
- beta0 = 2.497041993384243;
- projected PPO/tail-conflict repair;
- active tail descent kappa=.05;
- GAE lambda=.95;
- gamma=.99;
- one-model continuous preference schedule;
- original reward semantics;
- original reset/training distribution;
- adaptive critic support/head-refresh contract;
- optimizer states;
- actor learning rate;
- clipping;
- reset/seed schedule.

No:
- critic widening;
- Jacobian retention;
- rho/beta/gamma tuning;
- new replay states;
- new robustness intervention;
- semantic auxiliary loss.

## H2 continuation

Continue the compatible stack for 25 actor updates:

    global u30 -> u55

Save semantic-path snapshots at:

    u30
    u35
    u40
    u45
    u50
    u55

H2 training itself is the semantic accumulation experiment.

## Endpoint semantic contract

At every snapshot, use the exact frozen Foundation-V2 semantic endpoint protocol:

Matched reset suites:

    seeds 840001
          840002
          840003
          840004

Preferences:

    T = [.7,.1,.1,.1]
    A = [.1,.7,.1,.1]
    O = [.1,.1,.7,.1]
    S = [.1,.1,.1,.7]
    C = [.25,.25,.25,.25]

For each axis i compare heavy-i against matched center.

Objective channel:

    normalized objective-i heavy > center

Physical channel, lower is better:

    T: tracking_error
    A: ang_vel_xy
    O: tilt_deg
    S: action_rate

Endpoint axis PASS iff:

    objective-correct fraction >= .75
    physical-correct fraction  >= .75
    minimum survival           >= .95

No scalar objective/physical compensation is allowed.

## Timeline semantic quantities

For every snapshot report:
- endpoint PASS axes;
- simultaneous PASS count;
- objective heavy-center margins;
- physical heavy-center margins;
- minimum survival.

Across the path report:

### Acquisition

For each axis:

    first_pass_snapshot

if it ever passes.

### Retention after acquisition

For every axis that first passes, compute:

    retained fraction
      = fraction of later/equal snapshots that remain PASS

### PASS -> FAIL forgetting

Count every transition:

    PASS at snapshot k
      ->
    FAIL at snapshot k+1

Report:
- total PASS->FAIL events;
- per-axis events;
- PASS->FAIL rate among transitions whose source is PASS.

### FAIL -> PASS acquisition/recovery

Report analogously.

### Winner rotation

Winner rotation is present when:
- different axes alternate as PASS over consecutive checkpoints; and/or
- simultaneous competence fails to accumulate because acquired axes are lost as others are gained.

Report:
- PASS-set Jaccard between consecutive snapshots;
- number of transitions where pass_count does not increase but PASS identity changes;
- maximum simultaneous PASS count;
- final simultaneous PASS count.

## Primary H2 semantic accumulation gate

H2 semantic accumulation PASS requires all:

1. final endpoint PASS count = 4/4;
2. maximum simultaneous PASS count = 4/4;
3. every acquired axis retained fraction >= .80;
4. PASS->FAIL rate <= .20;
5. final semantic snapshot preserves survival >= .95;
6. final authority compatibility remains valid:
   - matched-support pairwise retention >= .90;
   - matched-support tangent retention >= .90;
   - every all-simplex edge-energy retention >= .90;
7. final critic compatibility remains valid:
   - H32 EV mean > 0;
   - MC64 EV mean > 0;
   - negative fractions <= .25.

This is intentionally stricter than “some axes improve.”

## Final continuum gate

Only the final u55 snapshot is evaluated on the frozen continuum contract.

Paths:

    T-A
    T-O
    T-S

Alphas:

    0, .25, .50, .75, 1

Matched continuum reset protocol is unchanged from Foundation V2.

Required:

    continuum monotonicity fraction >= .65
    endpoint-between fraction       >= .65

The continuum gate is conjunctive with endpoint accumulation.

## Interpretation

### H2 PASS

If endpoint accumulation + retention + continuum + authority/critic/survival all pass:

    preference authority durability
      is sufficient to support durable semantic accumulation
      in this tested regime.

This authorizes multi-seed confirmation.

### H2 FAIL — authority remains valid

If semantic accumulation fails while authority, critic, and survival remain valid:

    authority drift was a real mechanism,
    but not a complete explanation of semantic forgetting.

Do not reinterpret the failure as actor authority loss.

### H2 FAIL — authority compatibility regresses

If edge/tangent/critic/survival gates regress:

    H2 is confounded;
    semantic conclusion is blocked.

## Stop rule

No semantic-target training, ranking loss, Jacobian loss, critic widening, or robustness repair is authorized inside H2.

H2 is diagnostic of the stabilized stack.
