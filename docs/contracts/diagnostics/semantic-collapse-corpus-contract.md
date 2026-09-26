# Semantic Collapse Corpus Contract

Status: **FROZEN — DERIVED OPERATIONAL TAXONOMY FROM V15**

Date: 2026-09-24

## Purpose

Future causal analysis must not treat all binary PASS->FAIL events as equivalent.

Use two explicit terms:

### Robust semantic collapse
A held-out PASS->FAIL event satisfying all:

    boundary clearance >= 0.25 normalized units
    threshold-sweep robust flip = true
    exact paired-bootstrap PASS->FAIL probability >= 0.50

This is the primary causal corpus for future forgetting analysis.

### Boundary-sensitive semantic flip
A held-out PASS->FAIL event satisfying either:

    boundary clearance < 0.10

or

    threshold-sweep robust flip = false

These events belong primarily to evaluator-sensitivity / uncertainty analysis rather than controller-root-cause analysis.

### Other mixed
Events satisfying neither operational category remain separate and are not forced into either class.

## Scope

This taxonomy is derived entirely from the frozen v15 semantic-gate robustness audit.

It does NOT:
- change the frozen PASS/FAIL labels;
- change any semantic threshold;
- revise the v15 formal verdict;
- authorize a training method;
- authorize evaluator replacement.

## Corpus counts

From 14 held-out PASS->FAIL events:

    robust semantic collapse      8
    boundary-sensitive flip       5
    other mixed                   1

## Analysis rule

Any future statement about the mechanism of semantic forgetting should report:
1. result on robust semantic collapse corpus first;
2. boundary-sensitive events separately;
3. pooled PASS->FAIL only as a secondary compatibility statistic.

Binary PASS/FAIL should be accompanied by continuous gate margin whenever available.
