#ifndef INCLUDE_NN_NETWORKS_EXPANDING_LINEAR_FUNCTION_APPROXIMATOR_H_
#define INCLUDE_NN_NETWORKS_EXPANDING_LINEAR_FUNCTION_APPROXIMATOR_H_

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./network.h"

class ExpandingLinearFunctionApproximator : public Network {

 public:
  int no_of_active_input_features;
  float step_size;
  float meta_step_size;
  bool tidbd;

  ExpandingLinearFunctionApproximator(int no_of_input_features, int no_of_output_neurons, int no_of_active_input_features, float step_size, float meta_step_size, bool tidbd);

//  float introduce_targets(std::vector<float> targets);

  void set_input_values(std::vector<float> const &input_values);
  void step();
};

#endif // INCLUDE_NN_NETWORKS_LINEAR_FUNCTION_APPROXIMATOR_H_
