#!/usr/bin/env python3
"""Run the frozen FC-G Stage 1 branches sequentially on one GPU, then replay/probe."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
ROOT = REPO / "runs/teacher_v4_fc-2026-10-01/fcg"
PYTHON = Path(sys.executable)
SEEDS = {79101: 450, 79103: 350}
REPEATS = (11, 12, 13)
ARMS = ("GLOBAL", "SPLIT", "SPLIT-NORM-MATCHED")


def run(command, log):
    with log.open("w") as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)


def complete_train(directory, end):
    summary = directory / "summary.json"
    if not summary.exists() or not (directory / f"model_{end}.pt").exists():
        return False
    value = json.loads(summary.read_text())
    return value.get("iterations") == end


def complete_replay(directory, start):
    replay = directory / "replay"
    path = replay / "f2a_replay.json"
    if not path.exists() or not (replay / "rotation_traces.npz").exists():
        return False
    checkpoints = json.loads(path.read_text())["checkpoints"]
    return all(str(start + k) in checkpoints for k in (0, 10, 20, 30, 40, 50))


def main():
    for seed, start in SEEDS.items():
        origin = REPO / f"runs/teacher_v4_fb-2026-09-29/fb2a/seed{seed}/model_{start}.pt"
        end = start + 50
        for repeat in REPEATS:
            for arm in ARMS:
                directory = ROOT / f"seed{seed}" / f"{arm}_r{repeat}"
                directory.mkdir(parents=True, exist_ok=True)
                base = directory / f"model_{start}.pt"
                if not base.exists():
                    shutil.copy2(origin, base)
                print(f"START seed={seed} r={repeat} arm={arm}", flush=True)
                if not complete_train(directory, end):
                    run([str(PYTHON), "-u", str(HERE / "train_v4c.py"), "--out", str(directory),
                         "--resume", str(origin), "--iterations", str(end), "--save-every", "10",
                         "--seed", str(seed), "--branch-seed", str(repeat), "--num-envs", "4096",
                         "--objectives", "TAO", "--cardinalities", "1,2", "--lagrange-tmin", "0.52",
                         "--lagrange-lambda0", "0.786", "--lagrange-eta", "0.15", "--lagrange-cap", "20",
                         "--loss-arm", "full", "--gradient-composition", arm], directory / "train.out")
                print(f"TRAINED seed={seed} r={repeat} arm={arm}", flush=True)
                if not complete_replay(directory, start):
                    run([str(PYTHON), "-u", str(HERE / "f2a_bifurcation.py"), "--mode", "replay",
                         "--fold", "FCG", "--seed", str(seed), "--run-dir", str(directory),
                         "--out", str(directory / "replay"), "--traces"], directory / "replay.out")
                print(f"REPLAYED seed={seed} r={repeat} arm={arm}", flush=True)
    run([str(PYTHON), str(HERE / "fcg_probe.py"), "--stage", "1"], ROOT / "stage1_probe.out")
    print("STAGE1 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
