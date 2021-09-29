from abc import ABCMeta, abstractmethod


class BaseAgent:
    """Implements the agent class

    Note:
        train method is required.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def train(self, env, model, timesteps, epsilon, gamma, lmbda, logger, args):
        """Train the agent
        Args:
            _
        Return:
            -
        """
