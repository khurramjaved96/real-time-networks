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
                         int input_bins);

  void step();
  void imprint_LTU_randomly();
};

#endif // INCLUDE_NN_NETWORKS_IMPRINTING_ATARI_H_
