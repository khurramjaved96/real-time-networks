#ifndef INCLUDE_NN_NETWORKS_IMPRINTING_WIDE_H_
#define INCLUDE_NN_NETWORKS_IMPRINTING_WIDE_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class ImprintingWideNetwork: public Network {

 public:

  Neuron *bias_unit;
  float bound_replacement_prob;
  float bound_max_range;
  std::vector<BoundedNeuron *> bounded_neurons;

  ImprintingWideNetwork(int no_of_input_features,
                        int no_of_output_neurons,
                        int width_of_network,
                        std::vector<std::pair<float,float>> input_ranges,
                        float bound_replacement_prob,
                        float bound_max_range,
                        float step_size,
                        float meta_step_size,
                        bool tidbd,
                        int seed);

//  float introduce_targets(std::vector<float> targets);

  void step();
  std::vector<std::vector<std::pair<float, float>>> get_feature_bounds();
  std::vector<float> get_feature_utilities();
};

#endif // INCLUDE_NN_NETWORKS_IMPRINTING_WIDE_H_
