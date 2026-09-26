#!/usr/bin/env python3
"""Isaac plant snapshot of the scene exactly as configured."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.isaac_plant_snapshot import IsaacPlantSnapshot


class IsaacConfiguredPlantSnapshot(IsaacPlantSnapshot):
    """Isaac plant snapshot of the scene exactly as configured."""

    report = "isaac_snapshot.json"
    schema = "phase3_plant_isaac_snapshot_v1"


if __name__ == "__main__":
    IsaacConfiguredPlantSnapshot.main()
