import torch

from rl.core.modules.env_factor_encoder import EnvFactorEncoder


def test_forward_shape():
    encoder = EnvFactorEncoder(extrinsics_dim=9, latent_dim=8)
    e_t = torch.randn(4, 9)
    z_t = encoder(e_t)
    assert z_t.shape == (4, 8)


def test_gradients_flow_to_all_parameters():
    # Non-obvious requirement: mu must joint-train with the policy through
    # the SAME optimizer (RMA's architecture) — if a layer's gradient is
    # zero/None here, MOPPOTrainer's backward pass silently wouldn't train
    # it either.
    encoder = EnvFactorEncoder(extrinsics_dim=9, latent_dim=8)
    e_t = torch.randn(4, 9)
    z_t = encoder(e_t)
    z_t.sum().backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.any(param.grad != 0), f"{name} got an all-zero gradient"
