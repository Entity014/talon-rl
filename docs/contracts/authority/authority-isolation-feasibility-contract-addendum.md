# Authority-Isolation Feasibility Contract Addendum — Simplex-Tangent Jacobian Repair

Status: **PREDECLARED REPAIR OF INVALID GEOMETRY METRIC**
Date: 2026-09-25

## Reason for repair

The original feasibility contract evaluated the preference Jacobian as a full 4-D derivative with respect to `w`.

However the admissible preference domain is the simplex:

    w_i >= 0
    sum_i w_i = 1

Therefore only the 3-D tangent subspace satisfying:

    sum_i delta w_i = 0

is behaviorally meaningful.

A model may match the teacher exactly on the simplex while differing arbitrarily in the off-simplex normal direction. Penalizing that normal derivative is not a valid test of continuous preference-function preservation.

This issue was discovered because:
- held-out action imitation and preference separation were excellent;
- the full 4-D Jacobian metric reported a severe mismatch;
- finite differences along valid sum-zero directions showed close agreement.

## Repair

Replace only the two Jacobian criteria by their projection onto an orthonormal basis Q of the simplex tangent space:

    J_tan = J_w Q

where columns of Q span:

    {delta w in R^4 : 1^T delta w = 0}.

Use the same frozen thresholds:

    tangent Jacobian relative Frobenius error <= 15%
    tangent Jacobian cosine >= 0.95

No threshold, architecture, fit corpus, preference corpus, optimizer, fitting budget, or non-Jacobian metric is changed.

## Interpretation

The original full-4D Jacobian results remain recorded as off-simplex diagnostics but are invalid for the feasibility verdict.

The repaired tangent-Jacobian metrics determine criteria 6 and 7 only.

This is a geometry-contract correction, not a post-hoc threshold relaxation.
