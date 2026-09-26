import torch
from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo,vector_value_loss

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

def test_latent_action_logp_consistency():
    torch.manual_seed(7)
    m=T4SharedActorCritic(48,12)
    o=torch.randn(32,48);w=torch.full((32,4),.25)
    a,old,u=m.act_with_preference_latent(o,w)
    new=m.logp_from_pre_tanh_with_preference(o,w,u)
    ratio=torch.exp(new-old)
    assert float(a.abs().max()) <= 1.0
    assert torch.allclose(a,torch.tanh(u),atol=0,rtol=0)
    assert torch.allclose(new,old,atol=1e-6,rtol=0)
    assert torch.allclose(ratio,torch.ones_like(ratio),atol=1e-6,rtol=0)
