//
// Created by Khurram Javed on 2021-07-11.
//

#ifndef INCLUDE_NN_NETWORKS_ADAPTIVE_RECURRENT_NETWORK_H_
#define INCLUDE_NN_NETWORKS_ADAPTIVE_RECURRENT_NETWORK_H_



#include <vector>
#include <map>
#include <string>
#include <random>
#include "../dynamic_elem.h"
#include "../synapse.h"
#include "../neuron.h"
#include "./network.h"

class ContinuallyAdaptingRecurrentNetwork : public Network{


 public:


    ContinuallyAdaptingRecurrentNetwork(float step_size, int seed, int no_of_input_features);

    ~ContinuallyAdaptingRecurrentNetwork();

    void print_graph(Neuron *root);

    void step();

    float introduce_targets(std::vector<float> targets);

    float introduce_targets(std::vector<float> targets, float gamma, float lambda);

    float introduce_targets(std::vector<float> targets, float gamma, float lambda, std::vector<bool> no_grad);

    void add_feature(float step_size);
};


#endif  // INCLUDE_NN_NETWORKS_ADAPTIVE_RECURRENT_NETWORK_H_
