import numpy as np
from talon_rl.rewards.baselines import b0_reward

def test_tracking_shape_and_symmetry():
    z=.32
    assert np.isclose(b0_reward(.5,0,0,z,z,False),1)
    assert np.isclose(b0_reward(.25,0,0,z,z,False),b0_reward(.75,0,0,z,z,False))
    assert np.isclose(b0_reward(0,0,0,z,z,False),np.exp(-4))
    assert b0_reward(-.5,0,0,z,z,False) < .001

def test_penalties_and_single_terminal_charge():
    z=.32; base=b0_reward(.5,0,0,z,z,False)
    assert b0_reward(.5,.1,0,z,z,False) < base
    assert b0_reward(.5,0,0,z+.1,z,False) < base
    assert np.isclose(b0_reward(.5,0,0,z,z,True)-base,-10)
