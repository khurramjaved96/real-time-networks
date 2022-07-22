//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef INCLUDE_NN_NEURON_H_
#define INCLUDE_NN_NEURON_H_

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./message.h"
#include "./utils.h"

class Neuron : public dynamic_elem {
 public:
  static int64_t neuron_id_generator;
  static std::mt19937 gen;
  bool is_input_neuron;
  bool is_bias_unit;
  float value;
  float value_without_activation;
  float old_value;
  float old_old_value;
  float old_value_without_activation;
  int drinking_age;
  float shadow_error_prediction_before_firing;
  float shadow_error_prediction;
  float value_before_firing;
  int memory_made;
  float neuron_utility;
  float neuron_utility_to_distribute;
  float sum_of_utility_traces;
  bool is_output_neuron;
  bool useless_neuron;
  int sucesses;
  int failures;
  int64_t id;
  bool is_mature;
  int neuron_age;
  float average_activation;
  float mark_useless_prob;
  bool is_optical_flow_feature;
  int n_linear_synapses; // direct synapses (input->output)

  int n_successes;
  int n_failures;
  float activity_based_successes;
  float activity_based_failures;
  float imprinting_potential;


  std::pair<float, float> value_ranges;

  void forward_gradients();

  void update_value(int time_step);

  std::queue<message> error_gradient;
  std::queue<message_activation> past_activations;
  std::vector<synapse *> outgoing_synapses;
  std::vector<synapse *> incoming_synapses;

  int get_no_of_syanpses_with_gradients();

  Neuron(bool is_input, bool is_output);

  void fire(int time_step);

  float introduce_targets(float target, int timestep);

  float introduce_targets(float target, int timestep, float gamma, float lambda);

  float introduce_target(float td_error, float output_gradient, int timestep,  float gamma, float lambda);

  float introduce_targets(float target, int timestep, float gamma, float lambda, bool no_grad);

  void propagate_error();

//    Returns the gradient of the post activation w.r.t pre-activation
  virtual float backward(float output_grad) = 0;

  virtual float forward(float temp_value) = 0;

  virtual float backward_credit(float activation_value, synapse *it);

  void propagate_deep_error();

  void update_utility();

  void memory_leak_patch();

  void normalize_neuron();

  void mark_useless_weights();

  void mark_useless_linear_weights();

  void prune_useless_weights();

  void update_imprinting_potential();

  void update_synapse_contributions();

  ~Neuron() = default;
};

class Microstimuli : public Neuron {
 public:
  float backward(float output_grad);
  float forward(float temp_value);
  Microstimuli(bool is_input, bool is_output, float rate_of_change, float delay);

  float activation_threshold;
  float rate_of_change;
  float delay;
  float current_value;
  int current_timer;
  bool is_currently_active;
  bool is_currently_decreasing;
};

class LTU : public Neuron {
 public:
  float backward(float output_grad);
  float forward(float temp_value);
  LTU(bool is_input, bool is_output, float threshold);

  float activation_threshold;

};
class ReluNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  ReluNeuron(bool is_input, bool is_output);

};

class ReluThresholdNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  float threshold;

  ReluThresholdNeuron(float threshold);

};

class BiasNeuron : public Neuron {
 public:
  BiasNeuron();
  float backward(float output_grad);

  float forward(float temp_value);
};

class SigmoidNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  SigmoidNeuron(bool is_input, bool is_output);
};

class Tanh : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  Tanh(bool is_input, bool is_output);
};

class LinearNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  LinearNeuron(bool is_input, bool is_output);
};

class LeakyRelu : public Neuron {
  float negative_slope;
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  LeakyRelu(bool is_input, bool is_output, float negative_slope);
};

class BoundedNeuron: public Neuron {
 public:
  int num_times_reassigned;
  float bound_max_range;
  // an upper and a lower bound for each incoming synapse id
  std::unordered_map<int, std::pair<float, float>> activation_bounds;

  //TODO also should handle removal of synapses?
  void update_activation_bounds(synapse * incoming_synapse);
  void update_activation_bounds(synapse * incoming_synapse, float new_bound_center);

  float backward(float output_grad);

  float forward(float temp_value);

  BoundedNeuron(bool is_input, bool is_output, float bound_replacement_prob, float bound_max_range);

  void update_value(int time_step);
};

//class RecurrentNeuron : public Neuron {
// public:
//  virtual float backward_credit(float activation_value, synapse *);
//  RecurrentNeuron(bool is_input, bool is_output, synapse *recurrent_synapse);
//  ~RecurrentNeuron() = default;
//  bool is_recurrent_neuron;
//  synapse *recurrent_synapse;
//};
//
//class RecurrentReluNeuron : public RecurrentNeuron {
// public:
//  float backward(float output_grad);
//
//  float forward(float temp_value);
//
//  RecurrentReluNeuron(bool is_input, bool is_output, synapse *recurrent_synapse);
//
//};
//
//class RecurrentSigmoidNeuron : public RecurrentNeuron {
// public:
//  float backward(float output_grad);
//
//  float forward(float temp_value);
//
//  RecurrentSigmoidNeuron(bool is_input, bool is_output, synapse *recurrent_synapse);
//};


//
#endif  // INCLUDE_NN_NEURON_H_
