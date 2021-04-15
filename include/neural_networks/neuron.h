//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef BENCHMARKS_NEURON_H
#define BENCHMARKS_NEURON_H



#include <vector>
#include <queue>
#include <mutex>
#include <unordered_set>
#include "synapse.h"
#include "message.h"
#include <utility>

//class synapse;


class neuron {
    bool cycle_check;

public:
    static int neuron_id;
    float value;
    float temp_value;
    float temp_gradient;
    int depth;
    bool activation_type;
    int id;
    void forward_gradients();
    void update_value();
    std::queue<message> error_gradient;
    std::queue<std::pair<float, int>> past_activations;
    std::vector<synapse*> outgoing_synapses;
    std::vector<synapse*> incoming_synapses;
//    std::vector<std::reference_wrapper<neuron>> output_nodes;
    neuron(bool activation);
    void fire(int time_step);
    float introduce_targets(float target, int timestep);
    void propogate_error();
    void activation();
};

#endif //BENCHMARKS_NEURON_H