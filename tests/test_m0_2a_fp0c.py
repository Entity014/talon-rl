import torch
from rl.core.modules.actor_critic import ActorCritic
from rl.core.algorithms.vector_ppo import initialize_centered_from_scalar,W_REF
def test_centered_reference_has_zero_conditioning_gradient():
    torch.manual_seed(3);base=ActorCritic(8,8,3,1,[8,8]);m=initialize_centered_from_scalar(base,8,3,[8,8]);obs=torch.randn(16,8);w=torch.tensor(W_REF).expand(16,-1);loss=m.inference_centered(obs,w).pow(2).mean();loss.backward();assert torch.count_nonzero(m.actor_body[0].weight.grad[:,8:])==0
def test_centered_initial_function_matches_scalar_actor():
    torch.manual_seed(4);base=ActorCritic(8,8,3,1,[8,8]);m=initialize_centered_from_scalar(base,8,3,[8,8]);obs=torch.randn(16,8);w=torch.tensor(W_REF).expand(16,-1);assert torch.allclose(base.act_inference(obs),m.inference_centered(obs,w),atol=1e-6,rtol=1e-6)
