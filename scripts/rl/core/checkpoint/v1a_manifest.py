"""Deterministic provenance and formulation manifest for V1-A runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..modules.v1a_actor_critic import OBJECTIVES


V1A_EXCLUSIONS = {
    "vector_critic": False,
    "objective_specific_advantages": False,
    "preference_curriculum": False,
    "rehearsal": False,
    "anchor_loss": False,
    "learned_gate": False,
}


def canonical_hash(value: Any) -> str:
    """SHA-256 of a JSON value with stable key and float serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class V1AManifest:
    """Frozen run contract; hashes are intentionally separate."""

    manifest_version: str
    baseline_checkpoint: str
    baseline_normalization: str
    baseline_optimizer_state: str
    baseline_learned_std_state: str
    baseline_actor_architecture: dict[str, Any]
    action_distribution_semantics: str
    adapter_insertion_point: str
    adapter_activation: str
    objective_order: tuple[str, ...]
    w_ref: tuple[float, ...]
    bottleneck_dim: int
    parameter_budget_fraction: float
    inference_overhead_budget: float
    initialization: str
    evaluator_hash: str
    reset_suite_hash: str
    checkpoint_schema: str
    exclusion_flags: dict[str, bool] = field(default_factory=lambda: dict(V1A_EXCLUSIONS))
    source_baseline_hash: str = ""
    v1a_formulation_hash: str = ""

    def __post_init__(self) -> None:
        if tuple(self.objective_order) != OBJECTIVES:
            raise ValueError(f"objective order is frozen as {OBJECTIVES}")
        if len(self.w_ref) != len(OBJECTIVES) or any(value < 0 for value in self.w_ref):
            raise ValueError("w_ref must contain five non-negative objective weights")
        if abs(sum(self.w_ref) - 1.0) > 1e-6:
            raise ValueError("w_ref must lie on the simplex")
        if self.exclusion_flags != V1A_EXCLUSIONS:
            raise ValueError("all V1-B/C/D features must remain explicitly disabled")

    def formulation_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_baseline_hash", None)
        payload.pop("v1a_formulation_hash", None)
        return payload

    def with_hashes(self) -> "V1AManifest":
        source_payload = {
            "baseline_checkpoint": self.baseline_checkpoint,
            "baseline_normalization": self.baseline_normalization,
            "baseline_optimizer_state": self.baseline_optimizer_state,
            "baseline_learned_std_state": self.baseline_learned_std_state,
            "baseline_actor_architecture": self.baseline_actor_architecture,
        }
        return V1AManifest(
            **{
                **asdict(self),
                "source_baseline_hash": canonical_hash(source_payload),
                "v1a_formulation_hash": canonical_hash(self.formulation_payload()),
            }
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2) + "\n"
