import copy
import numpy as np


# There is problem with handling nan and inf values
# Ideally, we should use None but it cannot be sent through
# the interface. Maybe we should adjust add_values() to handle not passing
# the value to all the fields
INVALID_PLACEHOLDER = 1e+100


class LoggingManager:
    def __init__(self, log_to_db, run_id, model, commit_frequency=1000, episodic_metrics=None, neuron_metrics=None, synapse_metrics=None, prediction_metrics=None):
        self.log_to_db = log_to_db
        self.run_id = run_id
        self.model = model
        self.commit_frequency = commit_frequency

        self.episodic_metrics = episodic_metrics
        self.neuron_metrics = neuron_metrics
        self.synapse_metrics = synapse_metrics
        self.prediction_metrics = prediction_metrics

        self.episodic_log_vec = []
        self.neuron_log_vec = []
        self.synapse_log_vec = []
        self.prediction_log_vec = []

    def items_to_str(self, vec):
        ret_vec = []
        for v in vec:
            if type(v) == float and (np.isnan(v) or np.isinf(v)):
                ret_vec.append(str(INVALID_PLACEHOLDER))
            else:
                ret_vec.append(str(v))
        return ret_vec

    def commit_logs(self):
        if not self.log_to_db:
            return
        self.episodic_metrics.add_values(self.episodic_log_vec)
        self.neuron_metrics.add_values(self.neuron_log_vec)
        self.synapse_metrics.add_values(self.synapse_log_vec)
        self.prediction_metrics.add_values(self.prediction_log_vec)

        self.episodic_log_vec = []
        self.neuron_log_vec = []
        self.synapse_log_vec = []
        self.prediction_log_vec = []

    def log_step_metrics(self, episode, timestep):
        #if timestep % 5000 == 0:
        #    self.model.viz_graph()
        if not self.log_to_db:
            return
        max_vals = 20000
        # its so expensive to do a self.model.all_synapses or all_neurons at every step
        if timestep % 50000 == 0 and (len(self.model.all_synapses) > max_vals or len(self.model.all_neurons) > max_vals):
            print("Warning (logging_manager.py): Too many values to log! Truncating...")


        if timestep % 50000 == 0:
            for neuron in self.model.all_neurons[:max_vals]:
                self.neuron_log_vec.append(self.items_to_str([self.run_id, episode, timestep, neuron.id, neuron.value, neuron.average_activation, neuron.neuron_utility]))

        if timestep % 50000 == 0:
            for synapse in self.model.all_synapses[:max_vals]:
                self.synapse_log_vec.append(self.items_to_str([self.run_id, episode, timestep, synapse.id, synapse.weight, synapse.meta_step_size, synapse.synapse_utility]))

        if timestep % self.commit_frequency == 0:
            self.commit_logs()

    def log_eps_metrics(self, episode, timestep, MSRE, running_MSRE, error, predictions, return_target, return_error):
        if not self.log_to_db:
            return
        if episode % 100 == 0:
            self.episodic_log_vec.append(self.items_to_str([self.run_id, episode, timestep, MSRE, running_MSRE, error]))

        if episode % 500 == 0:
            for t, v in enumerate(zip(predictions, return_target, return_error)):
                self.prediction_log_vec.append(self.items_to_str([self.run_id, episode, t, MSRE, v[0], v[1], v[2], "[]"]))


