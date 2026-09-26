import torch

from rl.core.modules.actor_critic import ActorCritic
from rl.core.runtime.exporter import export_policy_as_jit


def test_act_inference_is_deterministic():
    """Unlike act() (samples from a Normal), act_inference() must give the
    exact same output for the same input every call — the property play.py
    and any exported deployment artifact depend on."""
    model = ActorCritic(actor_obs_dim=8, critic_obs_dim=8, action_dim=4, reward_dim=3, hidden_dims=[16, 16])
    obs = torch.randn(2, 8)

    out1 = model.act_inference(obs)
    out2 = model.act_inference(obs)

    assert torch.equal(out1, out2)


def test_exported_jit_module_matches_act_inference(tmp_path):
    """The exported JIT module must reproduce act_inference()'s output
    exactly for the same input — proves the export actually captured the
    trained actor weights, not just that torch.jit.save() didn't error."""
    model = ActorCritic(actor_obs_dim=8, critic_obs_dim=8, action_dim=4, reward_dim=3, hidden_dims=[16, 16])
    obs = torch.randn(3, 8)
    expected = model.act_inference(obs)

    export_path = str(tmp_path / "policy.pt")
    export_policy_as_jit(model, export_path)

    loaded = torch.jit.load(export_path)
    actual = loaded(obs)

    assert torch.allclose(actual, expected)
