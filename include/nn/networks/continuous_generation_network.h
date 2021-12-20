#ifndef INCLUDE_NN_NETWORKS_MICROSTIMULI_NETWORK_H_
#define INCLUDE_NN_NETWORKS_MICROSTIMULI_NETWORK_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include <utility>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class ContinuousGenerationNetwork: public Network {

 public:

  // TODO short_term features recycling ages U[min, max]
  Neuron *bias_unit;
  bool use_imprinting;
  float step_size;
  float meta_step_size;
  std::vector <int> input_indices;

  bool imprinting_only_single_layer;
  int imprinting_max_incoming_connections; // number of incoming connections per new feature= U[1:this value]
  int imprinting_max_microstimuli; // max number of microstimuli created per generation step= U[0:this value]
  int imprinting_max_new_features_per_step; // number of new non-microstimuli features created per generation step = U[1:this value]
  // number of steps that each short-term feature has before it gets marked as useless by
  // recycle_short_term_features() function = U[this_value.first, this_value.second]
  std::pair<int, int> short_term_feature_recycling_age;
  std::vector <Neuron*> linear_features; // features with input->output synapses
  std::vector <Neuron*> imprinted_features; // generated features that only need to pass the 2nd test on maturity
  std::vector <Neuron*> short_term_memory_features; // generated features that need to pass 1st test before maturity as well

  ContinuousGenerationNetwork(int no_of_input_features,
                      int no_of_output_neurons,
                      bool make_linear_connections,
                      float step_size,
                      float meta_step_size,
                      bool tidbd,
                      int seed,
                      bool use_imprinting,
                      int imprinting_max_incoming_connections,
                      int imprinting_max_num_microstimuli,
                      int imprinting_max_new_features_per_step,
                      bool imprinting_only_single_layer,
                      int linear_drinking_age,
                      float linear_synapse_local_utility_trace_decay,
                      float linear_utility_to_keep,
                      std::pair<int, int> short_term_feature_recycling_age);

  void step();
  void set_input_values(std::vector<float> const &input_values);
  void imprint_with_microstimuli();
  void recycle_short_term_memory_features(bool keep_subset, float perc_subset_to_keep);
};

#endif // INCLUDE_NN_NETWORKS_MICROSTIMULI_NETWORK_H_
