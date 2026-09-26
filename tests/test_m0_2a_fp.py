import torch
from rl.core.algorithms.vector_ppo import make_fp0_actor, fp0_probe

def test_fp0_initialization_preserves_action_and_logp():
    torch.manual_seed(7)
    base, expanded = make_fp0_actor(11, 4, [16, 16])
    obs = torch.randn(32, 11); w = torch.full((32, 5), 0.2)
    a0, a1, lp0, lp1 = fp0_probe(base, expanded, obs, w)
    assert torch.allclose(a0, a1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(lp0, lp1, atol=1e-6, rtol=1e-6)

def test_fp0_preference_columns_are_zero_and_critic_is_scalar():
    base, expanded = make_fp0_actor(8, 3, [8, 8])
    assert expanded.actor_body[0].weight.shape[1] == 13
    assert torch.count_nonzero(expanded.actor_body[0].weight[:, 8:]) == 0
    assert expanded.critic_head.out_features == base.critic_head.out_features == 1
