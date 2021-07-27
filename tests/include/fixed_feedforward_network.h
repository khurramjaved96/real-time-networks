//
// Created by Khurram Javed on 2021-04-25.
//

#ifndef INCLUDE_NEURAL_NETWORKS_NETWORKS_TEST_H_
#define INCLUDE_NEURAL_NETWORKS_NETWORKS_TEST_H_



#include <vector>
#include <map>
#include "../../include/nn/synapse.h"
#include "../../include/nn/neuron.h"
#include "../../include/nn/networks/network.h"

class TestCase : public Network{

 public:
    std::vector<float> sum_of_gradients;


    TestCase(float step_size, int width, int seed);

    TestCase(float step_size, int width, int seed, bool sigmoid_status);

    void step();
};



#endif  // INCLUDE_NEURAL_NETWORKS_NETWORKS_TEST_H_
