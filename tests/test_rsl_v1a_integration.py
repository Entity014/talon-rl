import torch
from torch import nn

from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
from rl.core.integration.rsl_rl.v1a_wrapper import RslRlV1AWrapper


class FakePolicy(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(4, 6), nn.ELU(), nn.Linear(6, 2))
        self.critic = nn.Sequential(nn.Linear(4, 6), nn.ELU(), nn.Linear(6, 1))
        self.std = nn.Parameter(torch.ones(2))
        self.distribution = None

    @property
    def action_std(self):
        return self.distribution.stddev


class FakeAlgorithm:
    def __init__(self):
        self.policy = FakePolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)


def test_attach_replaces_only_policy_and_preserves_scalar_ppo_optimizer_path(tmp_path):
    source = FakeAlgorithm()
    state = {"model_state_dict": source.policy.state_dict()}
    checkpoint = tmp_path / "m0_1.pt"
    torch.save(state, checkpoint)
    algorithm = FakeAlgorithm()
    wrapper = attach_v1a_policy(algorithm, checkpoint, bottleneck_dim=2)
    assert isinstance(algorithm.policy, RslRlV1AWrapper)
    assert algorithm.policy is wrapper
    assert algorithm.optimizer.param_groups[0]["lr"] == 1e-3
    assert len(list(algorithm.optimizer.param_groups[0]["params"])) > len(list(source.policy.parameters()))
