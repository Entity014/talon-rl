import itertools
import pytest
import torch

from talon_rl.config import ExtrinsicsCfg
from talon_rl.models.authority.teacher_v4 import TeacherV4


def _batch(b=3,m=4,seed=0):
    g=torch.Generator().manual_seed(seed)
    obs=torch.randn(b,48,generator=g);env=torch.randn(b,12,generator=g)
    ids=torch.arange(m).expand(b,-1).clone()
    w=torch.rand(b,m,generator=g);w=w/w.sum(-1,keepdim=True)
    return obs,env,ids,w


def test_env_dim_matches_extrinsics_contract():
    assert ExtrinsicsCfg().dim==TeacherV4().env_dim==12


def test_output_shapes():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch()
    a,lp,u=m.act(obs,env,ids,w)
    assert a.shape==u.shape==(3,12) and lp.shape==(3,)
    assert (a.abs()<=1).all()
    assert m.query_values(obs,env,ids,w).shape==(3,4)
    assert m.query_values(obs,env,ids,w,ids[:,:2]).shape==(3,2)


def test_logp_matches_sampled_logp():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch()
    _,lp,u=m.act(obs,env,ids,w)
    assert torch.allclose(m.logp_from_pre_tanh(obs,env,ids,w,u),lp,atol=1e-5)


def test_actor_and_critic_are_permutation_invariant():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch()
    a_ref=m.act_inference(obs,env,ids,w)
    v_ref=m.query_values(obs,env,ids,w,ids)
    for p in itertools.permutations(range(4)):
        p=torch.tensor(p)
        assert torch.allclose(m.act_inference(obs,env,ids[:,p],w[:,p]),a_ref,atol=1e-6)
        assert torch.allclose(m.query_values(obs,env,ids[:,p],w[:,p],ids),v_ref,atol=1e-5)


def test_zero_weight_entries_act_as_padding():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,_,_=_batch()
    ids=torch.tensor([[0,2]]*3);w=torch.tensor([[.6,.4]]*3)
    pad_ids=torch.tensor([[0,2,1,3]]*3);pad_w=torch.tensor([[.6,.4,0.,0.]]*3)
    assert torch.allclose(m.act_inference(obs,env,ids,w),m.act_inference(obs,env,pad_ids,pad_w),atol=1e-6)
    assert torch.allclose(m.query_values(obs,env,ids,w),m.query_values(obs,env,pad_ids,pad_w,ids),atol=1e-5)


@pytest.mark.parametrize("card",[1,2,3,4])
def test_variable_cardinality(card):
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch(m=card)
    assert m.act_inference(obs,env,ids,w).shape==(3,12)
    assert m.query_values(obs,env,ids,w).shape==(3,card)


def test_preference_and_env_change_action():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch()
    a=m.act_inference(obs,env,ids,w)
    assert not torch.allclose(a,m.act_inference(obs,env,ids,w.flip(-1)))
    assert not torch.allclose(a,m.act_inference(obs,env+1.0,ids,w))


def test_actor_critic_parameters_are_disjoint_and_complete():
    m=TeacherV4()
    actor={id(p) for p in m.actor_parameters()};critic={id(p) for p in m.critic_parameters()}
    assert actor and critic and not actor&critic
    assert actor|critic=={id(p) for p in m.parameters()}


def test_value_loss_does_not_touch_actor():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch()
    m.query_values(obs,env,ids,w).sum().backward()
    assert all(p.grad is None for p in m.actor_parameters())
    assert all(p.grad is not None for p in m.critic_parameters())


def test_policy_loss_does_not_touch_critic():
    torch.manual_seed(0);m=TeacherV4()
    obs,env,ids,w=_batch()
    _,_,u=m.act(obs,env,ids,w)
    m.logp_from_pre_tanh(obs,env,ids,w,u).sum().backward()
    assert all(p.grad is None for p in m.critic_parameters())


@pytest.mark.parametrize("ids,w",[
    (torch.tensor([[0,1]]),torch.tensor([[.5,.6]])),   # not normalized
    (torch.tensor([[0,4]]),torch.tensor([[.5,.5]])),   # id out of range
    (torch.tensor([[0,1]]),torch.tensor([[1.5,-.5]])), # negative weight
    (torch.tensor([[0,1]]).float(),torch.tensor([[.5,.5]])),
])
def test_invalid_objective_sets_raise(ids,w):
    m=TeacherV4();obs,env,_,_=_batch(b=1)
    with pytest.raises(ValueError):
        m.act_inference(obs,env,ids,w)
