import os

import numpy as np
import pytest

from rl.core.diagnostics.analyzer import Analyzer


def test_record_collects_requested_items_only():
    analyzer = Analyzer(items=["joint_vel"])
    analyzer.record({"joint_vel": np.zeros((4, 12)), "joint_torque": np.ones((4, 12))})
    analyzer.record({"joint_vel": np.ones((4, 12)), "joint_torque": np.ones((4, 12))})

    assert list(analyzer._data.keys()) == ["joint_vel"]
    assert len(analyzer._data["joint_vel"]) == 2


def test_record_raises_on_missing_key():
    analyzer = Analyzer(items=["does_not_exist"])
    with pytest.raises(KeyError):
        analyzer.record({"joint_vel": np.zeros((4, 12))})


def test_save_plots_writes_one_png_per_item(tmp_path):
    analyzer = Analyzer(items=["joint_vel", "obstacle_dist"])
    for t in range(5):
        analyzer.record({
            "joint_vel": np.full((3, 12), float(t)),
            "obstacle_dist": np.full(3, float(t)),  # 1-D per-lane signal
        })

    out_dir = str(tmp_path / "plots")
    analyzer.save_plots(out_dir)

    assert os.path.isfile(os.path.join(out_dir, "joint_vel.png"))
    assert os.path.isfile(os.path.join(out_dir, "obstacle_dist.png"))
