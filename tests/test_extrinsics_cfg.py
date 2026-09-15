import pytest

from talon_rl.config import ExtrinsicsCfg


def test_payload_dim_sums_mass_and_com_offset():
    cfg = ExtrinsicsCfg()
    assert cfg.payload_dim == cfg.payload_mass_dim + cfg.payload_com_offset_dim


def test_dim_includes_payload_under_explicit_observed_rewarded():
    cfg = ExtrinsicsCfg(payload_treatment="explicit_observed_rewarded")
    non_payload = (
        cfg.friction_dim + cfg.motor_power_scale_dim + cfg.leg_length_scale_dim
        + cfg.joint_range_scale_dim + cfg.terrain_height_dim
    )
    assert cfg.dim == non_payload + cfg.payload_dim


def test_dim_excludes_payload_under_noise_only():
    # noise_only: payload is randomized in sim but never observed by the
    # encoder — e_t's width itself shrinks, not just a reward-term toggle
    # (Pipeline_Summary.md §3.10: Baseline A "ไม่สังเกตหรือให้รางวัลแยก").
    cfg = ExtrinsicsCfg(payload_treatment="noise_only")
    non_payload = (
        cfg.friction_dim + cfg.motor_power_scale_dim + cfg.leg_length_scale_dim
        + cfg.joint_range_scale_dim + cfg.terrain_height_dim
    )
    assert cfg.dim == non_payload


def test_frozen_rejects_mutation():
    cfg = ExtrinsicsCfg()
    with pytest.raises(Exception):
        cfg.payload_mass_dim = 5
