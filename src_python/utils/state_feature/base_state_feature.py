
from abc import ABCMeta, abstractmethod

import numpy as np

class BaseStateFeature:
    """Implements the state feature transformer class

    Note:
        transform method is required.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        # This needs to be set by the subclass.
        self.feature_type = None
        pass

    @abstractmethod
    def transform(self, state):
        """Transforms the state into the state feature

            Args:
                state (Numpy array): the state observation

            Return:
                state_feature (Any): the state feature. This type should be reflected on self.feature_type on implementation
        """
    