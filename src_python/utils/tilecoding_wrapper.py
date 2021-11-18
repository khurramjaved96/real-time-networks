import copy
import numpy as np
from gym import spaces
from gym import ObservationWrapper

from .state_feature.state_feature_util import TileCoder


class TileCodedObservation(ObservationWrapper):
    """Return onehot tilecoded observations
    Args:
        env: The environment to wrap.
        ndims (int): number of dimensions in the state
        num_tilings (int): number of tilings to use
        ranges (list of tuples): (min,max) ranges for each dimension
        num_tiles (list of ints): number of tiles to use for each dimension
        is_expanding (bool): whether the observation shape will expand when new states are encountered
        expanding_initial_size (int): initial size of expanded features
    """

    def __init__(self, env, ndims, num_tilings, ranges, num_tiles, is_expanding=True, expanding_initial_size=20):
        super(TileCodedObservation, self).__init__(env)
        self._env = env
        self.is_expanding = is_expanding
        self.tc = TileCoder(ndims, num_tilings, ranges, num_tiles)

        if is_expanding:
            self.expanding_size = expanding_initial_size
        else:
            self.expanding_size = self.tc.total_tiles

    def observation(self, observation):
        indices = self.tc.transform(observation)
        curr_inp_size = max(indices) + 1
        if curr_inp_size > self.expanding_size:
            self.expanding_size = curr_inp_size
        onehot_obs = np.zeros(self.expanding_size)
        onehot_obs[indices] = 1
        return onehot_obs
