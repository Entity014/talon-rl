import torch
from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo,vector_value_loss

def test_shapes_and_simplex():
    m=T4SharedActorCritic(48,12)
    o=torch.randn(5,48);w=torch.full((5,4),.25)
    a,lp=m.act_with_preference(o,w);v=m.value_with_preference(o,w)
    assert a.shape==(5,12) and lp.shape==(5,) and v.shape==(5,4)
    try:m.act_with_preference(o,torch.ones(5,4))
    except ValueError:pass
    else:raise AssertionError("invalid simplex accepted")

def test_vector_gae_shapes():
    r=torch.randn(3,2,4);v=torch.randn(3,2,4);nv=torch.randn(2,4);d=torch.zeros(3,2,dtype=torch.bool)
    a,ret=vector_gae(r,v,nv,d);assert a.shape==r.shape and ret.shape==r.shape

def test_losses():
    ratio=torch.ones(6);adv=torch.randn(6,4);w=torch.full((6,4),.25)
    assert torch.isfinite(scalarized_late_weighted_ppo(ratio,adv,w))
    assert torch.isfinite(vector_value_loss(torch.zeros(6,4),torch.ones(6,4)))
