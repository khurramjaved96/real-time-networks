from abc import ABCMeta, abstractmethod

class BaseAgent:
    """Implements the agent class

    Note:
        train method is required.
    """

    __metaclass__ = ABCMeta

    def __init__(self, episodic_metric, neuron_metric, synapse_metric):
        self.episodic_metric = episodic_metric
        self.neuron_metric = neuron_metric
        self.synapse_metric = synapse_metric

        self.episodic_logger = []
        self.neuron_logger = []
        self.synapse_logger = []

    def log_network_metrics(self, model):
        pass

    def push_metrics(self):
        pass

    @abstractmethod
    def train(self, env, model, timesteps, epsilon, gamma, lmbda):
        """ Train the agent
            Args:
                _
            Return:
                -
        """
