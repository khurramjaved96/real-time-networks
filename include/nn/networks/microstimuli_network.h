#ifndef INCLUDE_NN_NETWORKS_MICROSTIMULI_NETWORK_H_
#define INCLUDE_NN_NETWORKS_MICROSTIMULI_NETWORK_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class MicrostimuliNetwork: public Network {

 public:

  Neuron *bias_unit;
  bool use_imprinting;
  int imprinting_start_timestep;
  float step_size;
  float meta_step_size;
  float imprinting_max_connection_prob; // max probability of an input being connected to a new feature = U[0:this value]
  int imprinting_max_incoming_connections; // max number of incoming connections a new feature can have
  int imprinting_max_microstimuli; // max number of microstimuli created for each feature= U[0:this value]
  bool imprinting_only_single_layer;
  float linear_utility_to_keep;
  std::vector <int> input_indices;
  std::vector <Neuron*> linear_features; // features with input->output synapses
  std::vector <Neuron*> imprinted_features; // generated features

  MicrostimuliNetwork(int no_of_input_features,
                      int no_of_output_neurons,
                      bool make_linear_connections,
                      float step_size,
                      float meta_step_size,
                      bool tidbd,
                      int seed,
                      bool use_imprinting,
                      float imprinting_max_connection_prob,
                      int imprinting_max_incoming_connections,
                      int imprinting_max_num_microstimuli,
                      bool imprinting_only_single_layer,
                      int linear_drinking_age,
                      float linear_synapse_local_utility_trace_decay,
                      float linear_utility_to_keep);

  void step();
  void set_input_values(std::vector<float> const &input_values);
  void imprint_with_microstimulus();
};

#endif // INCLUDE_NN_NETWORKS_MICROSTIMULI_NETWORK_H_
