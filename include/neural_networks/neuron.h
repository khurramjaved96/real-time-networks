//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef INCLUDE_NEURAL_NETWORKS_NEURON_H_
#define INCLUDE_NEURAL_NETWORKS_NEURON_H_


#include <vector>
#include <queue>
#include <unordered_set>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./message.h"
#include "./utils.h"

class neuron : public dynamic_elem {
 public:
    static int64_t neuron_id_generator;
    synapse *recurrent_synapse;
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
    int64_t id;
    bool is_mature;
    int neuron_age;
    float average_activation;

    void forward_gradients();

    void update_value();

    std::queue<message> error_gradient;
    std::queue<message_activation> past_activations;
    std::vector<synapse *> outgoing_synapses;
    std::vector<synapse *> incoming_synapses;

    explicit neuron(bool activation);

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


#endif  // INCLUDE_NEURAL_NETWORKS_NEURON_H_
