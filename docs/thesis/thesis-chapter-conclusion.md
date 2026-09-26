# Conclusion

This thesis investigated preference-conditioned multi-objective reinforcement learning for quadruped locomotion with an emphasis on semantic controllability rather than reward optimization alone.

The study first established a validated training and evaluation foundation. Objective semantics, PPO action/log-probability consistency, critic supervision, critic freshness, and matched semantic evaluation were repaired and independently checked before architecture-level conclusions were accepted. This foundation made it possible to distinguish genuine semantic limitations from implementation and evaluation confounds.

Under the validated foundation, direct preference conditioning was insufficient for the frozen four-objective semantic contract. Learned preference embeddings and single-site FiLM modulation increased preference-conditioned action authority and improved continuum interpolation, but complete semantic endpoint control did not emerge. V2-B was the strongest tested architecture in expressivity and interpolation, yet remained semantically incomplete.

Checkpoint analysis showed that semantic competencies could be acquired transiently and later lost during continued shared-policy optimization. This established semantic forgetting as a distinct problem rather than a simple failure of preference sensitivity or local learning signal.

The retention study then separated optimization stability from memory-content efficacy. Hard retention increasingly restricted plasticity. Soft rehearsal avoided hard constraints but could dominate the learning direction. A bounded auxiliary-gradient mechanism successfully prevented this takeover and preserved optimization geometry without collapsing policy plasticity.

However, stable rehearsal optimization was not sufficient for semantic retention. Preference-conditioned action-response memory did not produce a robust multi-seed advantage. Direct semantic-outcome rehearsal failed to provide a validated local gradient geometry, and deterministic semantic surrogates failed held-out seed generalization. A final literature-guided round using diversity-based trajectory memory and multi-timescale policy consolidation also failed the predeclared semantic short gate.

The principal conclusion is therefore:

> Stable retention optimization could be maintained without collapsing policy plasticity; however, neither the custom retention formulations nor the tested literature-guided diversity-replay and multi-timescale consolidation adaptations consistently preserved preference-conditioned semantic competence under the validated V2-B training setting.

The final reference configuration remains V2-B with Foundation V2, GAE λ = 0.95, and no retention intervention. This choice reflects the strongest validated architecture and training foundation without adding a retention mechanism whose superiority was not supported by the evidence.

The contributions of the study are threefold. First, it provides a controlled MORL training and evaluation framework that separates preference-conditioning behavior from reward, PPO, critic, and evaluation confounds. Second, it characterizes semantic acquisition and forgetting as distinct phenomena in a shared preference-conditioned locomotion policy. Third, it provides a systematic retention study showing that optimization-stability mechanisms and semantic-memory sufficiency must be evaluated separately.

Semantic retention therefore remains an open research problem in this setting, but no longer an unresolved implementation defect. The evidence narrows the future research question from how to stabilize rehearsal to how to represent and preserve behavior-level semantic competence across continued shared-policy optimization.
