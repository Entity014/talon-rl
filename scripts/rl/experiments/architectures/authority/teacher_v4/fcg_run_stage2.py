#!/usr/bin/env python3
"""Expand only FC-G seed × arm contrasts marked open by the frozen Stage 1 rule."""
import shutil

from fcg_run_stage1 import ARMS, HERE, PYTHON, REPO, ROOT, SEEDS, complete_replay, complete_train, run


def main():
    expansion = {line.strip() for line in (ROOT / "expand.txt").read_text().splitlines() if line.strip()}
    if not expansion or any("@" not in x or x.split("@")[0] not in ARMS[1:] or int(x.split("@")[1]) not in SEEDS for x in expansion):
        raise SystemExit(f"invalid or empty Stage 1 expansion: {sorted(expansion)}")
    for seed, start in SEEDS.items():
        open_arms = [arm for arm in ARMS[1:] if f"{arm}@{seed}" in expansion]
        if not open_arms:
            continue
        origin = REPO / f"runs/teacher_v4_fb-2026-09-29/fb2a/seed{seed}/model_{start}.pt"
        end = start + 50
        for repeat in (14, 15):
            for arm in ("GLOBAL", *open_arms):
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
    run([str(PYTHON), str(HERE / "fcg_probe.py"), "--stage", "2"], ROOT / "stage2_probe.out")
    print("STAGE2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
