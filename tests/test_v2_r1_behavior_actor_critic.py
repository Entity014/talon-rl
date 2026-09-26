import torch

from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
from talon_rl.models.behavior.latent import V2BehaviorActorCritic, initialize_from_v1c
from talon_rl.models.behavior.high_authority import FILM_AUTHORITY_R1, V2R1BehaviorActorCritic


def _post_film(model, obs, w, alpha):
    z = model.behavior_z(w)
    h = torch.nn.functional.elu(model.actor_pre(obs))
    return (1.0 + alpha * torch.tanh(model.film_gamma(z))) * h + alpha * torch.tanh(model.film_beta(z))


def test_r1_contract_only_changes_fixed_film_authority_and_state_schema():
    base = V2BehaviorActorCritic(6, 3, anchors=torch.randn(3, 2))
    r1 = V2R1BehaviorActorCritic(6, 3, anchors=base.behavior_anchors)
    assert FILM_AUTHORITY_R1 == 0.5
    assert set(base.state_dict()) == set(r1.state_dict())
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in r1.parameters())


def test_r1_identity_initialization_is_exactly_function_preserving():
    torch.manual_seed(41)
    obs_dim, action_dim = 6, 3
    source = V1CSharedActorCritic(obs_dim, action_dim)
    base = V2BehaviorActorCritic(obs_dim, action_dim)
    r1 = V2R1BehaviorActorCritic(obs_dim, action_dim)
    initialize_from_v1c(base, source.state_dict())
    initialize_from_v1c(r1, source.state_dict())

    obs = torch.randn(32, obs_dim)
    prefs = [
        torch.full((32, 3), 1 / 3),
        torch.tensor([[.8, .1, .1]]).repeat(32, 1),
        torch.tensor([[.1, .8, .1]]).repeat(32, 1),
        torch.tensor([[.1, .1, .8]]).repeat(32, 1),
    ]
    for w in prefs:
        with torch.no_grad():
            a_src = source.act_inference_with_preference(obs, torch.full_like(w, 1 / 3))
            a_base = base.act_inference_with_preference(obs, w)
            a_r1 = r1.act_inference_with_preference(obs, w)
            v_base = base.value_with_preference(obs, w)
            v_r1 = r1.value_with_preference(obs, w)
        assert torch.allclose(a_r1, a_base, atol=1e-6, rtol=1e-6)
        assert torch.allclose(a_r1, a_src, atol=1e-5, rtol=1e-5)
        assert torch.allclose(v_r1, v_base, atol=1e-6, rtol=1e-6)

        action = torch.tanh(torch.randn(32, action_dim)) * r1.ACTION_CLIP
        assert torch.allclose(
            r1.logp_with_preference(obs, w, action),
            base.logp_with_preference(obs, w, action),
            atol=1e-6,
            rtol=1e-6,
        )


def test_r1_same_weights_give_exactly_five_x_post_film_preference_delta():
    torch.manual_seed(9)
    anchors = torch.randn(3, 2)
    base = V2BehaviorActorCritic(7, 4, anchors=anchors)
    r1 = V2R1BehaviorActorCritic(7, 4, anchors=anchors)
    r1.load_state_dict(base.state_dict())

    # Create a nonzero learned-conditioning state while preserving identical weights.
    with torch.no_grad():
        base.behavior_encoder[2].weight.normal_(0, .2)
        base.behavior_encoder[2].bias.normal_(0, .05)
        base.film_gamma.weight.normal_(0, .2)
        base.film_gamma.bias.normal_(0, .05)
        base.film_beta.weight.normal_(0, .2)
        base.film_beta.bias.normal_(0, .05)
        r1.load_state_dict(base.state_dict())

    obs = torch.randn(24, 7)
    wp = torch.tensor([[.8, .1, .1]]).repeat(24, 1)
    wb = torch.tensor([[.1, .8, .1]]).repeat(24, 1)

    hp_base = _post_film(base, obs, wp, 0.1)
    hb_base = _post_film(base, obs, wb, 0.1)
    hp_r1 = _post_film(r1, obs, wp, 0.5)
    hb_r1 = _post_film(r1, obs, wb, 0.5)

    delta_base = hb_base - hp_base
    delta_r1 = hb_r1 - hp_r1
    assert torch.allclose(delta_r1, 5.0 * delta_base, atol=2e-6, rtol=2e-5)
