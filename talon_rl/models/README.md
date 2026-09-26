# Models

<!-- nav:start -->
[TALON package](../README.md)
<!-- nav:end -->


Neural-network architectures grouped by mechanism rather than experiment/version number.

- `foundations/`: shared preference-conditioned actor/critic foundations.
- `conditioning/`: mechanisms that inject preference authority into the policy.
- `behavior/`: explicit behavior-latent and projected behavior models.
- `authority/`: authority-isolated and objective-set-conditioned models.
- `auxiliary/`: training-only auxiliary heads.

Experiment identifiers such as T4, V1C, V2A, V2B, V2H, etc. remain in class names and experiment scripts for thesis traceability, not as filesystem boundaries.
