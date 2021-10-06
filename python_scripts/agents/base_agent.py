from abc import ABCMeta, abstractmethod


class BaseAgent:
    """Implements the agent class

    Note:
        train method is required.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.timestep = 0
        self.episode = 0
        self.MSRE = -1
        self.running_MSRE = -1

    @abstractmethod
    def train(self, env, model, timesteps, epsilon, gamma, lmbda, logger, args):
        """Train the agent
        Args:
            _
        Return:
            -
        """
