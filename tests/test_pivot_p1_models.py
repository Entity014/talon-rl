import torch
from talon_rl.optimization.scalar_critic import P1ScalarCriticActorCritic

def test_scalar_critic_has_one_value_output():
    m=P1ScalarCriticActorCritic(48,12);o=torch.zeros(4,48);w=torch.full((4,3),1/3)
    assert m.value_with_preference(o,w).shape==(4,1)
