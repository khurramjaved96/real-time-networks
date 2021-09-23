//
// Created by taodav on 29/7/21.
//

#ifndef FLEXIBLENN_SARSA_H
#define FLEXIBLENN_SARSA_H

#include "../../include/nn/networks/network.h"

class SarsaAgent {
  std::uniform_int_distribution<int> action_sampler;
  std::uniform_real_distribution<float> exploration_sampler;

  public:
    Network *network;
    int n_actions;
    float epsilon;
    float lambda;
    int steps;

    std::mt19937 mt;

    SarsaAgent(Network *in_network, int n_actions, float epsilon, float lambda);
    void set_eps(float eps);
    int step(std::vector<float> state);
    float post_step(int action, std::vector<float> next_state, float reward, float gamma);

};


#endif //FLEXIBLENN_SARSA_H
