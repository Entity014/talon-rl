"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_final_locomotion_score():
    """Run former final_locomotion_score.py stage."""
    """Score nine completed Final Locomotion Evaluation v1 shards."""
    
    
    import argparse
    import csv
    import json
    from collections import defaultdict
    from pathlib import Path
    
    import numpy as np
    
    
    FORWARD = (0.25, 0.5, 0.75)
    CONTRASTS = {
        "progress_heavy": ("vx_mae",),
        "efficiency_heavy": ("mean_mechanical_power_w", "energy_per_directed_m_jpm"),
        "impact_heavy": ("foot_force_p95_n",),
        "balance_heavy": ("tilt_rms_rad",),
    }
    
    
    def _load(paths: list[Path]) -> list[dict]:
        rows = []
        for path in paths:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    out = dict(row)
                    for key, value in row.items():
                        if key in ("preference",):
                            continue
                        out[key] = float(value)
                    out["source"] = str(path)
                    rows.append(out)
        return rows
    
    
    def _json_finite(value):
        """Represent undefined effects as JSON null, never non-standard NaN."""
        if isinstance(value, dict):
            return {key: _json_finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_finite(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    
    
    def _cell_gate(rows: list[dict]) -> dict:
        n = len(rows)
        successes = sum(r["episode_success"] == 1 for r in rows)
        falls = sum(r["fell"] == 1 for r in rows)
        return {
            "n": n, "successes": successes, "falls": falls,
            "success_fraction": successes / n if n else np.nan,
            "fall_fraction": falls / n if n else np.nan,
            "pass": n == 192 and successes >= 173 and falls <= 9,
        }
    
    
    def _paired_effect(uniform: list[dict], heavy: list[dict], metrics: tuple[str, ...]) -> dict:
        key = lambda r: (int(r["eval_seed"]), int(r["lane"]))
        u = {key(r): r for r in uniform if r["episode_success"] == 1}
        h = {key(r): r for r in heavy if r["episode_success"] == 1}
        paired = sorted(set(u) & set(h))
        effects = {}
        for metric in metrics:
            uv = np.array([u[k][metric] for k in paired], dtype=float)
            hv = np.array([h[k][metric] for k in paired], dtype=float)
            finite = np.isfinite(uv) & np.isfinite(hv) & (uv > 0)
            reduction = 1.0 - float(hv[finite].mean() / uv[finite].mean()) if finite.any() else np.nan
            effects[metric] = {
                "uniform_mean": float(uv[finite].mean()) if finite.any() else np.nan,
                "heavy_mean": float(hv[finite].mean()) if finite.any() else np.nan,
                "relative_reduction": reduction,
                "absolute_reduction": float((uv[finite] - hv[finite]).mean()) if finite.any() else np.nan,
            }
        return {"paired_successes": len(paired), "metrics": effects, "pairs": paired}
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("inputs", type=Path, nargs="+")
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        rows = _load(args.inputs)
        expected = 3 * 3 * 25 * 64
        if len(rows) != expected:
            raise SystemExit(f"need {expected} episode rows from nine full shards, got {len(rows)}")
    
        cells = defaultdict(list)
        for row in rows:
            cells[(int(row["training_seed"]), row["command_vx"], row["preference"])].append(row)
        # Old shards intentionally lack a repeated training_seed column only if
        # hand-edited; refuse them instead of inferring identity from filenames.
        if any("training_seed" not in row for row in rows):
            raise SystemExit("episode rows must contain training_seed")
    
        uniform = {}
        for seed in range(3):
            for vx in (-0.25, 0.0, 0.25, 0.5, 0.75):
                uniform[f"seed{seed}/vx{vx:g}"] = _cell_gate(cells[(seed, vx, "uniform")])
    
        contrast_results = {}
        for preference, metrics in CONTRASTS.items():
            seed_results = {}
            for seed in range(3):
                command_results = {}
                for vx in FORWARD:
                    u = cells[(seed, vx, "uniform")]
                    h = cells[(seed, vx, preference)]
                    paired = _paired_effect(u, h, metrics)
                    gate = _cell_gate(h)
                    improves = all(paired["metrics"][m]["relative_reduction"] >= 0.10 for m in metrics)
                    if preference == "progress_heavy":
                        improves &= paired["metrics"]["vx_mae"]["absolute_reduction"] >= 0.005
                    guard = True
                    if preference == "balance_heavy":
                        extra = _paired_effect(u, h, ("mean_abs_vz_mps",))["metrics"]["mean_abs_vz_mps"]
                        guard = extra["relative_reduction"] >= -0.05
                        paired["guard_mean_abs_vz"] = extra
                    if preference == "impact_heavy":
                        extra = _paired_effect(u, h, ("foot_force_max_n",))["metrics"]["foot_force_max_n"]
                        guard = extra["relative_reduction"] >= -0.05
                        paired["guard_max_foot_force"] = extra
                    command_results[str(vx)] = {
                        "locomotion_gate": gate,
                        "paired": {k: v for k, v in paired.items() if k != "pairs"},
                        "improves_10pct": bool(improves), "guard_pass": bool(guard),
                        "demonstrated": bool(gate["pass"] and paired["paired_successes"] >= 154 and improves and guard),
                        "not_worse_5pct": bool(
                            gate["pass"] and paired["paired_successes"] >= 154 and guard
                            and all(paired["metrics"][m]["relative_reduction"] >= -0.05 for m in metrics)
                        ),
                    }
                demonstrated = sum(x["demonstrated"] for x in command_results.values())
                remaining_ok = all(x["demonstrated"] or x["not_worse_5pct"] for x in command_results.values())
                seed_results[f"seed{seed}"] = {
                    "commands": command_results,
                    "pass": demonstrated >= 2 and remaining_ok,
                }
            contrast_results[preference] = {
                "training_seeds": seed_results,
                "result": "PASS" if all(x["pass"] for x in seed_results.values()) else "NOT-DEMONSTRATED",
            }
    
        result = {
            "criteria_label": "project-defined milestone criteria",
            "episode_rows": len(rows),
            "uniform_cells": uniform,
            "locomotion_result": "PASS" if all(x["pass"] for x in uniform.values()) else "FAIL",
            "preference_contrasts": contrast_results,
        }
        result["overall_result"] = "ACCEPTABLE" if (
            result["locomotion_result"] == "PASS"
            and all(x["result"] == "PASS" for x in contrast_results.values())
        ) else "INADEQUATE"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"locomotion": result["locomotion_result"], "overall": result["overall_result"]}))
    
    
    if True:
        main()

def run_run_final_locomotion_shards():
    """Run former run_final_locomotion_shards.py stage."""
    """Run/resume all nine GPU shards for Final Locomotion Evaluation v1."""
    
    
    import argparse
    import csv
    import os
    import subprocess
    import sys
    from pathlib import Path
    
    
    ROOT = Path(__file__).resolve().parents[4]
    
    
    def complete(path: Path) -> bool:
        if not path.is_file():
            return False
        with path.open(newline="") as stream:
            return sum(1 for _ in csv.reader(stream)) == 1601
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/final_locomotion_eval_v1/episodes")
        args = parser.parse_args()
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'scripts'}"
        python = Path("/home/xero/isaac-lab-env/bin/python")
        runner = ROOT / "scripts/rl/experiments/common/utilities/final_locomotion_eval.py"
        for training_seed in range(3):
            checkpoint = ROOT / f"runs/phase1_hipact_dt01_seed{training_seed}_2026-09-20/checkpoints/checkpoint.pt"
            for eval_seed in (1001, 1002, 1003):
                output = args.output_dir / f"seed{training_seed}_eval{eval_seed}_episodes.csv"
                if complete(output):
                    print(f"skip complete shard seed={training_seed} eval={eval_seed}", flush=True)
                    continue
                print(f"start shard seed={training_seed} eval={eval_seed}", flush=True)
                subprocess.run(
                    [str(python), str(runner), str(checkpoint), "--training-seed", str(training_seed),
                     "--eval-seed", str(eval_seed), "--output-dir", str(args.output_dir)],
                    cwd=ROOT, env=env, check=True,
                )
                if not complete(output):
                    raise RuntimeError(f"shard did not produce 1600 episode rows: {output}")
        print("all nine shards complete", flush=True)
    
    
    if True:
        main()

def run_run_final_locomotion_shards_r1():
    """Run former run_final_locomotion_shards_r1.py stage."""
    """Run/resume all nine GPU shards for Final Locomotion Evaluation v1,
    rerun against the R1 checkpoints (artifacts/r1_freeze/FREEZE.md). Same
    protocol, same thresholds, same script (final_locomotion_eval.py) as the
    frozen v1 baseline run -- only the checkpoint set and output directory
    differ. See run_final_locomotion_shards.py (the original, left untouched,
    still points at the phase1_hipact_dt01_* baseline checkpoints)."""
    
    
    import csv
    import os
    import subprocess
    from pathlib import Path
    
    
    ROOT = Path(__file__).resolve().parents[4]
    
    
    def complete(path: Path) -> bool:
        if not path.is_file():
            return False
        with path.open(newline="") as stream:
            return sum(1 for _ in csv.reader(stream)) == 1601
    
    
    def main() -> None:
        output_dir = ROOT / "artifacts/final_locomotion_eval_r1/episodes"
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'scripts'}"
        python = Path("/home/xero/isaac-lab-env/bin/python")
        runner = ROOT / "scripts/rl/experiments/common/utilities/final_locomotion_eval.py"
        for training_seed in range(3):
            checkpoint = ROOT / f"runs/phase1_r1_dt01_seed{training_seed}_2026-09-20/checkpoints/checkpoint.pt"
            for eval_seed in (1001, 1002, 1003):
                output = output_dir / f"seed{training_seed}_eval{eval_seed}_episodes.csv"
                if complete(output):
                    print(f"skip complete shard seed={training_seed} eval={eval_seed}", flush=True)
                    continue
                print(f"start shard seed={training_seed} eval={eval_seed}", flush=True)
                subprocess.run(
                    [str(python), str(runner), str(checkpoint), "--training-seed", str(training_seed),
                     "--eval-seed", str(eval_seed), "--output-dir", str(output_dir)],
                    cwd=ROOT, env=env, check=True,
                )
                if not complete(output):
                    raise RuntimeError(f"shard did not produce 1600 episode rows: {output}")
        print("all nine shards complete", flush=True)
    
    
    if True:
        main()

STAGES = {
    "final_locomotion_score": run_final_locomotion_score,
    "run_final_locomotion_shards": run_run_final_locomotion_shards,
    "run_final_locomotion_shards_r1": run_run_final_locomotion_shards_r1,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
