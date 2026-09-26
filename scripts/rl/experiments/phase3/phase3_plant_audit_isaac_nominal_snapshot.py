#!/usr/bin/env python3
"""Isaac plant snapshot with base-mass randomisation switched off."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.isaac_plant_snapshot import IsaacPlantSnapshot


class IsaacNominalPlantSnapshot(IsaacPlantSnapshot):
    """Isaac plant snapshot of the nominal model, not one sampled instance."""

    report = "isaac_nominal_snapshot.json"
    schema = "phase3_plant_isaac_nominal_snapshot_v1"

    def configure(self, cfg):
        cfg.events.add_base_mass = None


if __name__ == "__main__":
    IsaacNominalPlantSnapshot.main()
