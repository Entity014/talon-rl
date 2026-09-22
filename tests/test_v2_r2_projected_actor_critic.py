import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from talon_rl.v1c_actor_critic import V1CSharedActorCritic
from talon_rl.v2_r2_projected_actor_critic import V2R2ProjectedActorCritic, initialize_from_v1c

def basis():
    q,_=torch.linalg.qr(torch.randn(128,2))
    return q.T.contiguous()

def test_r2_basis_and_identity_coefficients():
    b=basis();m=V2R2ProjectedActorCritic(6,3,b,projection_alpha=1.0)
    w=torch.tensor([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]],dtype=torch.float32)
    assert torch.allclose(m.semantic_basis@m.semantic_basis.T,torch.eye(2),atol=1e-5)
    assert torch.equal(m.semantic_coefficients(w),torch.zeros(3,2))
    assert torch.equal(m.projected_delta(w),torch.zeros(3,128))

def test_r2_function_preserving_init_action_value_logp():
    torch.manual_seed(9);obs_dim,ad=6,3
    src=V1CSharedActorCritic(obs_dim,ad);m=V2R2ProjectedActorCritic(obs_dim,ad,basis(),projection_alpha=1.0)
    initialize_from_v1c(m,src.state_dict())
    obs=torch.randn(16,obs_dim);w=torch.full((16,3),1/3);action=torch.tanh(torch.randn(16,ad))*src.ACTION_CLIP
    with torch.no_grad():
        assert torch.allclose(src.act_inference_with_preference(obs,w),m.act_inference_with_preference(obs,w),atol=1e-5,rtol=1e-5)
        assert torch.allclose(src.value_with_preference(obs,w),m.value_with_preference(obs,w),atol=1e-5,rtol=1e-5)
    assert torch.allclose(src.logp_with_preference(obs,w,action),m.logp_with_preference(obs,w,action),atol=1e-5,rtol=1e-5)

def test_r2_checkpoint_contains_basis_and_coeff_state():
    b=basis();m=V2R2ProjectedActorCritic(4,3,b)
    restored=V2R2ProjectedActorCritic(4,3,basis())
    restored.load_state_dict(m.state_dict())
    assert torch.allclose(restored.semantic_basis,m.semantic_basis)
    assert torch.allclose(restored.coeff_net[2].weight,m.coeff_net[2].weight)

def test_r2_manifold_gradient_finite():
    m=V2R2ProjectedActorCritic(4,3,basis(),anchors=torch.randn(3,2))
    w=torch.tensor([[.8,.1,.1],[.1,.8,.1]])
    loss=m.manifold_loss(w);assert torch.isfinite(loss);loss.backward()
    assert m.behavior_encoder[0].weight.grad is not None
