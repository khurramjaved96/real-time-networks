//
// Created by Khurram Javed on 2021-08-12.
//

#ifndef INCLUDE_NN_NETWORKS_LINEAR_FUNCTION_APPROXIMATOR_H_
#define INCLUDE_NN_NETWORKS_LINEAR_FUNCTION_APPROXIMATOR_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class LinearFunctionApproximator : public Network {

 public:

  LinearFunctionApproximator(int no_of_input_features, int no_of_output_neurons, float step_size, float meta_step_size, bool tidbd);

//  float introduce_targets(std::vector<float> targets);

  void step();
};

#endif // INCLUDE_NN_NETWORKS_LINEAR_FUNCTION_APPROXIMATOR_H_
