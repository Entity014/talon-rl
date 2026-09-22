import torch

from talon_rl.v1c_actor_critic import V1CSharedActorCritic
from talon_rl.v2_behavior_actor_critic import V2BehaviorActorCritic, initialize_from_v1c


def test_v2_z_reference_is_barycentric_and_finite():
    anchors = torch.tensor([[-1., 0.], [1., 0.], [0., 2.]])
    model = V2BehaviorActorCritic(4, 3, anchors=anchors)
    w = torch.tensor([[.8, .1, .1], [.1, .8, .1], [1 / 3, 1 / 3, 1 / 3]])
    assert torch.allclose(model.z_reference(w), w @ anchors)
    assert torch.isfinite(model.behavior_z(w)).all()


def test_v2_identity_film_and_v1_reference_initialization_preserve_action_and_logp():
    torch.manual_seed(4)
    obs_dim, action_dim = 6, 3
    source = V1CSharedActorCritic(obs_dim, action_dim)
    target = V2BehaviorActorCritic(obs_dim, action_dim)
    initialize_from_v1c(target, source.state_dict())
    obs = torch.randn(12, obs_dim)
    w = torch.full((12, 3), 1 / 3)
    with torch.no_grad():
        a_src = source.act_inference_with_preference(obs, w)
        a_tgt = target.act_inference_with_preference(obs, w)
    assert torch.allclose(a_src, a_tgt, atol=1e-5, rtol=1e-5)
    action = torch.tanh(torch.randn(12, action_dim)) * target.ACTION_CLIP
    assert torch.allclose(source.logp_with_preference(obs, w, action), target.logp_with_preference(obs, w, action), atol=1e-5, rtol=1e-5)


def test_v2_manifold_gradient_and_checkpoint_state():
    model = V2BehaviorActorCritic(4, 3, anchors=torch.randn(3, 2))
    w = torch.tensor([[.8, .1, .1], [.1, .8, .1]])
    loss = model.manifold_loss(w)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.behavior_encoder[0].weight.grad is not None
    restored = V2BehaviorActorCritic(4, 3)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored.behavior_anchors, model.behavior_anchors)
