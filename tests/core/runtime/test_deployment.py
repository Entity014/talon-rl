import numpy as np
import pytest
import torch

from rl.core.runtime.deployment import PolicyRuntime, PolicyRuntimeConfig


class FixedPolicy(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.cat((observation[:, :1] * 4.0, observation[:, 1:2] * 4.0), dim=1)


def _runtime(tmp_path):
    path = tmp_path / "policy.pt"
    torch.jit.script(FixedPolicy()).save(str(path))
    return PolicyRuntime(
        str(path), PolicyRuntimeConfig(observation_dim=2, action_dim=2, action_low=-1.0, action_high=1.0)
    )


def test_runtime_clips_actions_at_policy_boundary(tmp_path):
    runtime = _runtime(tmp_path)

    action = runtime.act(np.array([[0.5, -0.5]], dtype=np.float32))

    assert np.array_equal(action, np.array([[1.0, -1.0]], dtype=np.float32))


def test_runtime_rejects_wrong_observation_width(tmp_path):
    runtime = _runtime(tmp_path)

    with pytest.raises(ValueError, match="observation width"):
        runtime.act(np.zeros((1, 3), dtype=np.float32))


def test_runtime_rejects_non_finite_observation(tmp_path):
    runtime = _runtime(tmp_path)

    with pytest.raises(ValueError, match="non-finite"):
        runtime.act(np.array([[np.nan, 0.0]], dtype=np.float32))
