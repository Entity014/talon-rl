import os

import pytest

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg

from rl.core.run_dir import dump_config, make_run_dir, resolve_checkpoint


def test_make_run_dir_with_explicit_name(tmp_path):
    run_dir = make_run_dir(str(tmp_path), run_name="my_run")
    assert run_dir == os.path.join(str(tmp_path), "my_run")
    assert os.path.isdir(run_dir)


def test_make_run_dir_defaults_to_a_timestamp(tmp_path):
    run_dir = make_run_dir(str(tmp_path))
    name = os.path.basename(run_dir)
    assert len(name) == len("2026-09-15_14-30-00")  # %Y-%m-%d_%H-%M-%S
    assert os.path.isdir(run_dir)


def test_dump_config_writes_yaml_with_one_key_per_cfg(tmp_path):
    run_dir = make_run_dir(str(tmp_path), run_name="run1")
    dump_config(run_dir, obs=ObservationSpaceCfg(), action=ActionSpaceCfg())

    config_path = os.path.join(run_dir, "config.yaml")
    assert os.path.isfile(config_path)

    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    assert set(data.keys()) == {"obs", "action"}
    assert data["action"]["dim"] == ActionSpaceCfg().dim
    assert data["obs"]["joint_pos_dim"] == ObservationSpaceCfg().joint_pos_dim


def test_resolve_checkpoint_exact_run_name(tmp_path):
    run_dir = make_run_dir(str(tmp_path), run_name="run1")
    open(os.path.join(run_dir, "checkpoint.pt"), "w").close()

    path = resolve_checkpoint(str(tmp_path), "run1")
    assert path == os.path.join(run_dir, "checkpoint.pt")


def test_resolve_checkpoint_last_picks_most_recently_modified(tmp_path):
    import time

    run1 = make_run_dir(str(tmp_path), run_name="run1")
    open(os.path.join(run1, "checkpoint.pt"), "w").close()
    time.sleep(0.01)
    run2 = make_run_dir(str(tmp_path), run_name="run2")
    open(os.path.join(run2, "checkpoint.pt"), "w").close()

    path = resolve_checkpoint(str(tmp_path), "last")
    assert path == os.path.join(run2, "checkpoint.pt")


def test_resolve_checkpoint_raises_when_missing(tmp_path):
    make_run_dir(str(tmp_path), run_name="run1")  # no checkpoint.pt written
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path), "run1")
