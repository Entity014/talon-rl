"""Frozen V1-C preference curriculum sampler.

The JSON manifest is the source of truth for stage ranges, mixtures,
Dirichlet concentrations, and the evaluation grid.  This module contains no
training logic and is intentionally deterministic given a NumPy Generator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = ROOT / "artifacts" / "v1c" / "V1C_CURRICULUM_MANIFEST.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text())


def stage_for_update(update: int, manifest: dict | None = None) -> dict:
    if manifest is None:
        manifest = load_manifest()
    if update < 1 or update > manifest["common"]["updates"]:
        raise ValueError(f"update must be in [1, {manifest['common']['updates']}]")
    for stage in manifest["conditions"]["curriculum"]["stages"]:
        if stage["updates"][0] <= update <= stage["updates"][1]:
            return stage
    raise RuntimeError(f"no curriculum stage covers update {update}")


def _region_concentration(region: str, manifest: dict) -> np.ndarray:
    spec = manifest["sampler_regions"][region]
    return np.asarray(spec["concentration"], dtype=np.float64)


def _draw_region(rng: np.random.Generator, region: str, count: int, manifest: dict) -> np.ndarray:
    if count == 0:
        return np.empty((0, len(manifest["common"]["objective_order"])), dtype=np.float32)
    return rng.dirichlet(_region_concentration(region, manifest), size=count).astype(np.float32)


def sample_preferences(
    rng: np.random.Generator,
    update: int,
    num_envs: int,
    condition: str,
    manifest: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one simplex preference per environment and return region labels."""
    if manifest is None:
        manifest = load_manifest()
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if condition not in {"curriculum", "full_simplex_control"}:
        raise ValueError("condition must be curriculum or full_simplex_control")

    if condition == "full_simplex_control":
        labels = np.full(num_envs, "full_simplex", dtype="U32")
    else:
        stage = stage_for_update(update, manifest)
        mixture = dict(stage["mixture"])
        labels = rng.choice(
            np.asarray(list(mixture), dtype="U32"),
            size=num_envs,
            p=np.asarray(list(mixture.values()), dtype=np.float64),
        )
        if "objective_heavy_union" in labels:
            heavy = np.asarray(["progress_heavy", "balance_heavy", "efficiency_heavy"], dtype="U32")
            heavy_probs = np.asarray(list(stage["objective_heavy_union"].values()), dtype=np.float64)
            labels[labels == "objective_heavy_union"] = rng.choice(heavy, size=int(np.sum(labels == "objective_heavy_union")), p=heavy_probs)

    samples = np.empty((num_envs, len(manifest["common"]["objective_order"])), dtype=np.float32)
    for region in np.unique(labels):
        mask = labels == region
        samples[mask] = _draw_region(rng, str(region), int(mask.sum()), manifest)
    if not np.isfinite(samples).all() or (samples < 0).any():
        raise FloatingPointError("preference sampler produced invalid values")
    samples /= samples.sum(axis=1, keepdims=True)
    return samples.astype(np.float32), labels


def evaluation_grid(manifest: dict | None = None) -> np.ndarray:
    if manifest is None:
        manifest = load_manifest()
    grid = np.asarray(manifest["evaluation_grid"]["points"], dtype=np.float32)
    if grid.ndim != 2 or not np.allclose(grid.sum(axis=1), 1.0, atol=1e-7) or (grid < 0).any():
        raise ValueError("manifest evaluation grid is not a simplex grid")
    return grid
