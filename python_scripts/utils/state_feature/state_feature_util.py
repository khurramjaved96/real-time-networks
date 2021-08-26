from enum import Enum

from numpy.core.fromnumeric import resize
from .base_state_feature import BaseStateFeature
from .tile_coding import *
import numpy as np


class StateFeatureType(Enum):
    INDEX = "index"
    VECTOR = "vector"


class TileCoder(BaseStateFeature):
    def __init__(self, ndims, num_tilings, ranges, num_tiles):
        """Initialization

        Args:
            ndims (int): number of dimensions in the state
            num_tilings (int): number of tilings to use
            ranges (list of tuples): (min,max) ranges for each dimension
            num_tiles (list of ints): number of tiles to use for each dimension

        """
        assert ndims == len(ranges) == len(
            num_tiles), "Number of dimensions, length of ranges, and length of num tiles must match"

        self.feature_type = StateFeatureType.INDEX

        self.ndims = ndims
        self.num_tilings = num_tilings
        self.ranges = ranges
        self.num_tiles = num_tiles

        # Get total number of tiles
        tiles_per_tiling = 1
        for tiles in num_tiles:
            tiles_per_tiling *= tiles + 1

        self.total_tiles = tiles_per_tiling * num_tilings

        self.iht = IHT(self.total_tiles)
        pass

    def transform(self, state):
        """Transforms the state into the state feature

        Args:
            state (Numpy array): the state observation
        Returns:
            state_feature (Numpy array): A list of indices where the feature vector is activated.
        """

        if (self.ndims != len(state)):
            raise TypeError("Number of dimension on state doesn't match the dimension on tile coder")

        resized_state = []
        for ind, v in enumerate(state):
            new_v = (v - self.ranges[ind][0]) / ((self.ranges[ind]
                                                  [1] - self.ranges[ind][0])) * self.num_tiles[ind]
            resized_state.append(new_v)
        indices = tiles(self.iht, self.num_tilings, resized_state)
        return indices

class IdentityFeature(BaseStateFeature):
    def __init__(self):
        pass

    def transform(self, state):
        return state

