from talon_rl.tasks.manipulation.tienkung_env.config import (
    ActionSpaceCfg,
    ObservationSpaceCfg,
    RewardVectorCfg,
)


def test_observation_space_total_dim_sums_all_fields():
    cfg = ObservationSpaceCfg()
    assert cfg.total_dim == (
        cfg.joint_pos_dim
        + cfg.joint_vel_dim
        + cfg.box_relative_pos_dim
        + cfg.arm_contact_dim
        + cfg.prev_action_dim
        + cfg.preference_dim
    )
    assert cfg.total_dim == 34


def test_action_space_dim_matches_arm_dof():
    assert ActionSpaceCfg().dim == 8


def test_reward_vector_cfg_dim_matches_term_count():
    cfg = RewardVectorCfg()
    assert cfg.term_names == ("progress", "clearance", "energy", "impact", "smoothness")
    assert cfg.dim == 5
    assert len(cfg.active) == 5
    assert all(cfg.active)
