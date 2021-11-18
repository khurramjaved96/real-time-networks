import copy
import numpy as np


# There is problem with handling nan and inf values
# Ideally, we should use None but it cannot be sent through
# the interface. Maybe we should adjust add_values() to handle not passing
# the value to all the fields
INVALID_PLACEHOLDER = 1e+100


class LoggingManager:
    def __init__(self, log_to_db, run_id, model, commit_frequency=1000, episodic_metrics=None, neuron_metrics=None, synapse_metrics=None, prediction_metrics=None, bounded_unit_metrics=None, imprinting_metrics=None, linear_feature_metrics=None):
        self.log_to_db = log_to_db
        self.run_id = run_id
        self.model = model
        self.commit_frequency = commit_frequency

        self.episodic_metrics = episodic_metrics
        self.neuron_metrics = neuron_metrics
        self.synapse_metrics = synapse_metrics
        self.prediction_metrics = prediction_metrics
        self.bounded_unit_metrics = bounded_unit_metrics
        self.imprinting_metrics = imprinting_metrics
        self.linear_feature_metrics = linear_feature_metrics

        self.episodic_log_vec = []
        self.neuron_log_vec = []
        self.synapse_log_vec = []
        self.prediction_log_vec = []
        self.bounded_log_vec = []
        self.imprinting_log_vec = []
        self.linear_feature_log_vec = []

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
        try:
            if self.episodic_metrics:
                self.episodic_metrics.add_values(self.episodic_log_vec)
            if self.neuron_metrics:
                self.neuron_metrics.add_values(self.neuron_log_vec)
            if self.synapse_metrics:
                self.synapse_metrics.add_values(self.synapse_log_vec)
            if self.prediction_metrics:
                self.prediction_metrics.add_values(self.prediction_log_vec)
            if self.bounded_unit_metrics:
                self.bounded_unit_metrics.add_values(self.bounded_log_vec)
            if self.imprinting_metrics:
                self.imprinting_metrics.add_values(self.imprinting_log_vec)
            if self.linear_feature_metrics:
                self.linear_feature_metrics.add_values(self.linear_feature_log_vec)
        except:
            print("Failed commitinng logs")

        self.episodic_log_vec = []
        self.neuron_log_vec = []
        self.synapse_log_vec = []
        self.prediction_log_vec = []
        self.bounded_log_vec = []
        self.imprinting_log_vec = []
        self.linear_feature_log_vec = []

    def log_step_metrics(self, episode, timestep):
        #if timestep % 5000 == 0:
        #    self.model.viz_graph()
        if not self.log_to_db:
            return
        max_vals = 100000
        # its so expensive to do a self.model.all_synapses or all_neurons at every step
        if self.neuron_metrics or self.synapse_metrics:
            if timestep % 100000 == 0 and (len(self.model.all_synapses) > max_vals or len(self.model.all_neurons) > max_vals):
                print("Warning (logging_manager.py): Too many values to log! Truncating...")


        if self.neuron_metrics is not None:
            if timestep % 100000 == 0:
                for neuron in self.model.all_neurons[:max_vals]:
                    self.neuron_log_vec.append(self.items_to_str([self.run_id, episode, timestep, neuron.id, neuron.value, neuron.average_activation, neuron.neuron_utility]))

        if self.synapse_metrics is not None:
            if timestep % 100000 == 0:
                for synapse in self.model.all_synapses[:max_vals]:
                    self.synapse_log_vec.append(self.items_to_str([self.run_id, episode, timestep, synapse.id, synapse.weight, synapse.step_size, synapse.synapse_utility]))

        if timestep % self.commit_frequency == 0:
            self.commit_logs()

    def log_eps_metrics(self, episode, timestep, MSRE, running_MSRE, error, predictions, return_target, return_error):
        if timestep% 1000 == 0:
            print(">>","T:", timestep, "\t\tEps:", episode, "\t\tMSRE:",  MSRE, "\t\trunning_MSRE:", running_MSRE, "\t\tgenerated features:", len(self.model.imprinted_features), "\t\tSynapses:", len(self.model.all_synapses), "\t\tPred[-1]:", predictions[-1])
        #if episode% 20 == 0:
        #    print(predictions)
        #    print(return_target)
        if not self.log_to_db:
            return
        if self.episodic_metrics is not None:
            if timestep % 1000 == 0:
                self.episodic_log_vec.append(self.items_to_str([self.run_id, episode, timestep, MSRE, running_MSRE, error]))

        if self.prediction_metrics is not None:
            if timestep % 2500 == 0:
                for t, v in enumerate(zip(predictions, return_target, return_error)):
                    self.prediction_log_vec.append(self.items_to_str([self.run_id, episode, t, MSRE, v[0], v[1], v[2], "[]"]))

    def log_synapse_replacement(self, bound_replacement_metrics):
        return
        if not self.log_to_db:
            return
        bound_replacement_vec = []
        replaced_neurons = self.model.get_reassigned_bounded_neurons()
        try:
            for n in replaced_neurons:
                bound_replacement_vec.append(self.items_to_str([self.run_id, n.id, n.neuron_age, n.neuron_utility, n.outgoing_synapses[0].weight, n.num_times_reassigned]))
            bound_replacement_metrics.add_values(bound_replacement_vec)
        except:
            print("Syn replacement log failed")

    def log_bounded_unit_activity(self, episode, timestep):
        return
        if not self.log_to_db:
            return
        if self.bounded_unit_metrics is not None:
            if episode % 1000 == 0:
                self.bounded_log_vec.append(self.items_to_str([self.run_id, episode, timestep, self.model.count_active_bounded_units()]))

    def log_imprinting_activity(self, episode, timestep):
        if not self.log_to_db:
            return
        if self.imprinting_metrics is None:
            return
        # this gets too big, want to update only once per commit
        if len(self.imprinting_log_vec) != 0:
            return
        for imprinted_unit in self.model.imprinted_features:
            for s in imprinted_unit.incoming_synapses:
                if len(imprinted_unit.outgoing_synapses) < 1:
                    continue
                self.imprinting_log_vec.append(self.items_to_str([self.run_id, episode, timestep, imprinted_unit.id, s.input_neuron.id, imprinted_unit.outgoing_synapses[0].weight, imprinted_unit.outgoing_synapses[0].step_size, imprinted_unit.neuron_age, imprinted_unit.neuron_utility]))

    def log_linear_feature_activity(self, episode, timestep):
        if not self.log_to_db:
            return
        if self.linear_feature_metrics is None:
            return
        # this gets too big, want to update only once per commit
        if len(self.linear_feature_log_vec) != 0:
            return
        for linear_unit in self.model.linear_features:
            for s in linear_unit.outgoing_synapses:
                if s.output_neuron.is_output_neuron:
                    self.linear_feature_log_vec.append(self.items_to_str([self.run_id, episode, timestep, linear_unit.id, s.weight, s.step_size, linear_unit.neuron_utility, s.synapse_utility, s.synapse_utility_to_distribute]))
