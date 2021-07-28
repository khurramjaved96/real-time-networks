//
// Created by Khurram Javed on 2021-07-20.
//

#ifndef INCLUDE_NEURAL_NETWORKS_NETWORKS_TEST_RECURRENT_H_
#define INCLUDE_NEURAL_NETWORKS_NETWORKS_TEST_RECURRENT_H_

#include <string>
#include <vector>
#include <map>
#include <random>
#include "../../include/nn/dynamic_elem.h"
#include "../../include/nn/synapse.h"
#include "../../include/nn/neuron.h"
#include "../../include/nn/networks/network.h"

class ContinuallyAdaptingRecurrentNetworkTest : public Network {
 public:
  ContinuallyAdaptingRecurrentNetworkTest(float step_size, int seed, int no_of_input_features);

  ~ContinuallyAdaptingRecurrentNetworkTest();

  void add_feature(float step_size);

  void step();
};

#endif  // INCLUDE_NEURAL_NETWORKS_NETWORKS_TEST_RECURRENT_H_
