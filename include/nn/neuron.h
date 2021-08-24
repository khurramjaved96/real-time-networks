//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef INCLUDE_NN_NEURON_H_
#define INCLUDE_NN_NEURON_H_

#include <vector>
#include <queue>
#include <unordered_set>
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
  float value;
  float value_without_activation;
  float old_value;
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

  void prune_useless_weights();

  ~Neuron() = default;
};

class ReluNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  ReluNeuron(bool is_input, bool is_output);

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
