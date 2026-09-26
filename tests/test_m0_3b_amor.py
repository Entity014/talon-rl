import torch
from rl.core.algorithms.amor import RunningObservationNorm, ScaleAlignedAmorActorCritic

def test_running_norm_roundtrip_and_freeze():
    n = RunningObservationNorm(3); x = torch.randn(20,3); n.update(x); before=n.mean.clone(); n.freeze(); n.update(torch.randn(20,3)); assert torch.equal(before,n.mean)
    state=n.state_dict(); extra=n.state_dict_extra(); m=RunningObservationNorm(3); m.load_state_dict(state); m.load_state_dict_extra(extra); assert torch.allclose(m.mean,n.mean) and m.frozen

def test_scale_model_shapes():
    m=ScaleAlignedAmorActorCritic(11,4,[8,8,8,8]); o=torch.randn(5,11); w=torch.full((5,5),.2); a,lp=m.act_with_preference(o,w); assert a.shape==(5,4) and m.value_with_preference(o,w).shape==(5,5) and torch.isfinite(lp).all()
