import copy
import numpy as np
from gym import spaces
from gym import ObservationWrapper

from .state_feature.state_feature_util import TileCoder


class BinnedObservation(ObservationWrapper):
    """Return binned image observations
    Args:
        env: The environment to wrap.
        nbins (int): number of bins to divide the input values
    """

    def __init__(self, env, nbins):
        super(BinnedObservation, self).__init__(env)
        self._env = env
        self.nbins = nbins
        self.checked_dims = False
        self.unwrapped_obs = None

    def observation(self, observation):
        if not self.checked_dims:
            assert observation.shape[0] == 1, f"Env might not be supported, do check"
            assert observation.shape[3] == 4, f"Env might not be supported, do check"
            self.checked_dims = True
        self.unwrapped_obs = observation
        # get 4th obs
        latest_obs = observation[0,:,:,-1]
        # [H,W] with values indicating bin index
        bins_assignments = np.digitize(latest_obs, np.arange(0,255,255/self.nbins))
        # [nbins,H,W] binary
        binned_obs = np.array([ 1*(bin_assignments==i+1) for i in range(self.nbins) ])
        return binned_obs
