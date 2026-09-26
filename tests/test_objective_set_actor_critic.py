import itertools
import torch

from talon_rl.models.authority.objective_set import (
    ObjectiveSetAuthorityIsolatedWideCritic,
    canonical_tokens, aggregate_objective_set, set_from_dense_preference,
)

def test_weighted_set_context_matches_dense_preference():
    w=torch.tensor([[.7,.1,.1,.1],[.25,.25,.25,.25]],dtype=torch.float32)
    tok,ww=set_from_dense_preference(w)
    z=aggregate_objective_set(tok,ww)
    assert torch.allclose(z,w,atol=0,rtol=0)

def test_set_aggregation_is_permutation_invariant():
    w=torch.tensor([[.55,.15,.2,.1]],dtype=torch.float32)
    tok,ww=set_from_dense_preference(w)
    ref=aggregate_objective_set(tok,ww)
    for p in itertools.permutations(range(4)):
        q=torch.tensor(p)
        got=aggregate_objective_set(tok[:,q],ww[:,q])
        assert torch.allclose(got,ref,atol=1e-7,rtol=0)

def test_actor_and_token_query_critic_are_permutation_invariant():
    torch.manual_seed(0)
    m=ObjectiveSetAuthorityIsolatedWideCritic(48,12)
    obs=torch.randn(3,48)
    w=torch.tensor([[.7,.1,.1,.1],[.25,.25,.25,.25],[.1,.7,.1,.1]])
    tok,ww=set_from_dense_preference(w)
    ar=m.act_inference_from_set(obs,tok,ww)
    vr=m.query_values_from_set(obs,tok,ww,tok)
    p=torch.tensor([2,0,3,1])
    ap=m.act_inference_from_set(obs,tok[:,p],ww[:,p])
    vp=m.query_values_from_set(obs,tok[:,p],ww[:,p],tok)
    assert torch.allclose(ap,ar,atol=1e-6,rtol=0)
    assert torch.allclose(vp,vr,atol=1e-6,rtol=0)
def test_variable_cardinality_uses_same_model_shapes():
    torch.manual_seed(1)
    m=ObjectiveSetAuthorityIsolatedWideCritic(48,12)
    obs=torch.randn(2,48)
    base=canonical_tokens()
    for ids in ([0,1],[0,2,3],[0,1,2,3]):
        ids=torch.tensor(ids)
        tok=base[ids].unsqueeze(0).repeat(2,1,1)
        ww=torch.full((2,len(ids)),1.0/len(ids))
        a=m.act_inference_from_set(obs,tok,ww)
        v=m.query_values_from_set(obs,tok,ww,tok)
        assert a.shape==(2,12)
        assert v.shape==(2,len(ids))
        assert torch.isfinite(a).all() and torch.isfinite(v).all()
