#ifndef INCLUDE_NN_NETWORKS_IMPRINTING_ATARI_H_
#define INCLUDE_NN_NETWORKS_IMPRINTING_ATARI_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class ImprintingAtariNetwork: public Network {

 public:

  Neuron *bias_unit;
  bool use_imprinting;
  float step_size;
  float meta_step_size;
  int input_H;
  int input_W;
  int input_bins;
  float imprinting_max_prob;
  bool imprinting_only_single_layer;
  bool use_optical_flow_state;
  std::vector <int> input_indices;
  std::vector <Neuron*> linear_features; // input->output synapses
  std::vector <Neuron*> imprinted_features;

  ImprintingAtariNetwork(int no_of_input_features,
                         int no_of_output_neurons,
                         int width_of_network,
                         float step_size,
                         float meta_step_size,
                         bool tidbd,
                         int seed,
                         bool use_imprinting,
                         int input_H,
                         int input_W,
                         int input_bins,
                         float imprinting_max_prob,
                         bool imprinting_only_single_layer,
                         bool use_optical_flow_state);

  void step();
  void set_input_values(std::vector<float> const &input_values);
  void imprint_on_interesting_neurons(std::vector<Neuron *> interesting_neurons);
  void imprint_randomly();
  void imprint_using_optical_flow();
  void imprint_using_optical_flow_old();
};

#endif // INCLUDE_NN_NETWORKS_IMPRINTING_ATARI_H_
