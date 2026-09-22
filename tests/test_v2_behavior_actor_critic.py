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
    assert torch.allclose(source.value_with_preference(obs, w), target.value_with_preference(obs, w), atol=1e-5, rtol=1e-5)


def test_v2_r1_alpha_half_preserves_exact_identity_at_initialization():
    torch.manual_seed(7)
    obs_dim, action_dim = 6, 3
    source = V1CSharedActorCritic(obs_dim, action_dim)
    base = V2BehaviorActorCritic(obs_dim, action_dim, film_alpha=0.1)
    r1 = V2BehaviorActorCritic(obs_dim, action_dim, film_alpha=0.5)
    initialize_from_v1c(base, source.state_dict())
    initialize_from_v1c(r1, source.state_dict())
    obs = torch.randn(16, obs_dim)
    w = torch.full((16, 3), 1 / 3)
    action = torch.tanh(torch.randn(16, action_dim)) * source.ACTION_CLIP
    with torch.no_grad():
        a_src = source.act_inference_with_preference(obs, w)
        a_base = base.act_inference_with_preference(obs, w)
        a_r1 = r1.act_inference_with_preference(obs, w)
        v_src = source.value_with_preference(obs, w)
        v_base = base.value_with_preference(obs, w)
        v_r1 = r1.value_with_preference(obs, w)
    assert torch.allclose(a_src, a_base, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_src, a_r1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_src, v_base, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_src, v_r1, atol=1e-5, rtol=1e-5)
    lp_src = source.logp_with_preference(obs, w, action)
    assert torch.allclose(lp_src, base.logp_with_preference(obs, w, action), atol=1e-5, rtol=1e-5)
    assert torch.allclose(lp_src, r1.logp_with_preference(obs, w, action), atol=1e-5, rtol=1e-5)


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
