//
// Created by taodav on 17/9/21.
//

#ifndef FLEXIBLENN_ADAPTIVE_VALUE_NETWORK_H
#define FLEXIBLENN_ADAPTIVE_VALUE_NETWORK_H

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class AdaptiveValueNetwork : public Network {
  Neuron *bias_unit;

  public:
    AdaptiveValueNetwork(int no_of_input_features, int no_output_neurons, int seed,
                         float step_size, float meta_step_size, bool tidbd,
                         float utility_to_keep);

    void step();
    void add_feature(float step_size, float utility_to_keep);
};


#endif //FLEXIBLENN_ADAPTIVE_VALUE_NETWORK_H
