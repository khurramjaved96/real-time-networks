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
#include "utils.h"
#include <utility>

//class synapse;


class neuron {
    bool cycle_check;

public:
    static int neuron_id;
    bool input_neuron;
    static normal_random normal_dist;
    float value;
    float temp_value;
    int memory_made;
    bool activation_type;
    bool output_neuron;
    bool useless_neuron;
    int id;
    float average_activation;
    void forward_gradients();
    void update_value();
    std::queue<message> error_gradient;
    std::queue<std::pair<float, int>> past_activations;
    std::vector<synapse*> outgoing_synapses;
    std::vector<synapse*> incoming_synapses;
//    std::vector<std::reference_wrapper<neuron>> output_nodes;
    neuron(bool activation);
    neuron(bool activation, bool output_n);
    neuron(bool activation, bool output_n, bool input_n) ;
    void fire(int time_step);
    float introduce_targets(float target, int timestep);
    float introduce_targets(float target, int timestep, float gamma, float lambda);
    void propogate_error();
    void mark_useless_weights();
    void prune_useless_weights();
    void activation();
    void init_incoming_synapses();
};

#endif //BENCHMARKS_NEURON_H