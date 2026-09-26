"""Tests for scalar-critic pivot primitives."""

# -----------------------------------------------------------------------------
# Former: test_pivot_p1_gae.py
# -----------------------------------------------------------------------------

import torch
from talon_rl.optimization.scalar_critic import control_advantage, normalize_final_advantage, scalar_gae, treatment_advantage, vector_gae_local

def test_scalar_gae_matches_one_objective_vector_gae():
    r=torch.tensor([[[1.,2.,3.]],[[.5,1.,-1.]]]);v=torch.zeros_like(r);nv=torch.zeros_like(r[0]);d=torch.zeros((2,1),dtype=torch.bool);w=torch.ones_like(r);w[:,:,1:]=0
    a_s,_=scalar_gae(r[:,:,0],v[:,:,0],nv[:,0],d);a_v,_=vector_gae_local(r,v,nv,d)
    assert torch.allclose(a_s,a_v[:,:,0])

def test_control_and_treatment_reduce_to_same_path_when_objectives_share_a_value_scale():
    r=torch.tensor([[[1.,2.,3.]],[[.5,1.,-1.]]]);v=torch.tensor([[[.1,.2,.3]],[[.2,.1,.0]]]);nv=torch.tensor([[.2,.1,.0]]);d=torch.zeros((2,1),dtype=torch.bool);w=torch.tensor([[[.2,.3,.5]],[[.2,.3,.5]]])
    # Identical objective signals make scalarization commute with GAE.
    r=r*torch.tensor([1.,1.,1.]);v=v*torch.tensor([1.,1.,1.]);ac,_=control_advantage(r,w,v.sum(-1),nv.sum(-1),d);at,_=treatment_advantage(r,w,v,nv,d)
    assert torch.allclose(ac,at,atol=1e-6)

def test_final_normalization_is_zero_mean_unit_variance():
    x=normalize_final_advantage(torch.tensor([1.,2.,3.,4.]));assert abs(float(x.mean()))<1e-6;assert abs(float(x.std(unbiased=False))-1)<1e-6

def test_bootstrap_is_masked_on_termination():
    r=torch.zeros((2,1));v=torch.zeros((2,1));nv=torch.ones((1,));d=torch.tensor([[True],[False]]);a,_=scalar_gae(r,v,nv,d);assert torch.allclose(a[0],torch.zeros(1))

# -----------------------------------------------------------------------------
# Former: test_pivot_p1_models.py
# -----------------------------------------------------------------------------

import torch
from talon_rl.optimization.scalar_critic import P1ScalarCriticActorCritic

def test_scalar_critic_has_one_value_output():
    m=P1ScalarCriticActorCritic(48,12);o=torch.zeros(4,48);w=torch.full((4,3),1/3)
    assert m.value_with_preference(o,w).shape==(4,1)
