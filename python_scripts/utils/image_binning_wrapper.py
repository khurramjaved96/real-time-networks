import copy
import numpy as np
import gym
from gym import spaces
from gym import ObservationWrapper

from .state_feature.state_feature_util import TileCoder


class BinnedObservation(ObservationWrapper):
    """Return binned image observations
    Didnt work out with the gym obs wrapped due to atari preprocessing
    Args:
        env: The environment to wrap.
        nbins (int): number of bins to divide the input values
    """

    def __init__(self, env, nbins, generate_optical_flow=False):
        env.reward_range = (-1,1)
        super(BinnedObservation, self).__init__(env)
        self.env = env
        self.nbins = nbins
        self.generate_optical_flow = generate_optical_flow
        self.checked_dims = False
        self.unwrapped_obs = None
        self.binned_obs = None

    def observation(self, observation):
        if not self.checked_dims:
            assert observation.shape[0] == 1, f"Env might not be supported, do check"
            assert observation.shape[3] == 4, f"Env might not be supported, do check"
            self.checked_dims = True
        self.unwrapped_obs = observation
        # get 4th obs
        latest_obs = observation[0,:,:,-1]
        # [H,W] with values indicating bin index
        bin_assignments = np.digitize(latest_obs, np.arange(0,255,255/self.nbins))
        # [nbins,H,W] binary
        self.binned_obs = np.array([ 1*(bin_assignments==i+1) for i in range(self.nbins) ])
        if self.generate_optical_flow:
            diff = observation[0,:,:,-2] != observation[0,:,:,-1]
            if np.sum(diff) > 500:
                # happens when entire screen is reset
                print("Optical flow vector too large, skipping it")
                diff = diff * 0
            self.binned_obs = np.concatenate((self.binned_obs, np.expand_dims(diff*1, axis=0)), axis=0)
        return self.binned_obs.flatten()
