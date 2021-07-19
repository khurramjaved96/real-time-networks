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
#include "dynamic_elem.h"


class neuron : public dynamic_elem {
public:
    static long long int neuron_id_generator;
    synapse* recurrent_synapse;
    float old_value;
    bool is_recurrent_neuron;
    bool is_input_neuron;
    float value;
    int drinking_age;
    float shadow_error_prediction_before_firing;
    float shadow_error_prediction;
    float value_before_firing;
    int memory_made;
    float neuron_utility;
    bool is_relu;
    bool is_output_neuron;
    bool useless_neuron;
    int sucesses;
    int failures;
    long long int id;
    bool is_mature;
    int neuron_age;
    float average_activation;

    void forward_gradients();

    void update_value();

    std::queue<message> error_gradient;
    std::queue<message_activation> past_activations;
    std::vector<synapse *> outgoing_synapses;
    std::vector<synapse *> incoming_synapses;

    neuron(bool activation);

    neuron(bool activation, bool output_n);

    neuron(bool activation, bool output_n, int id);

    neuron(bool activation, bool output_n, bool input_n);

    void fire(int time_step);

    float introduce_targets(float target, int timestep);

    float introduce_targets(float target, int timestep, float gamma, float lambda);

    void propagate_error();

    void update_utility();

    void mark_useless_weights();

    void prune_useless_weights();

    ~neuron() = default;
};


//class recurrent_neuron : public neuron {
//public:
//
//
//    synapse *recurrent_synapse;
//
//    recurrent_neuron(bool activation, bool output_n, bool input_n);
//
////    void fire(int time_step);
//
//    float introduce_targets(float target, int timestep, float gamma, float lambda);
//
//    void propagate_error();
//
//    void mark_useless_weights();
//
//    void prune_useless_weights();
//
//    ~recurrent_neuron() = default;
//
//};

#endif //BENCHMARKS_NEURON_H