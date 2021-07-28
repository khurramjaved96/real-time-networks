//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef INCLUDE_NN_NEURON_H_
#define INCLUDE_NN_NEURON_H_


#include <vector>
#include <queue>
#include <unordered_set>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./message.h"
#include "./utils.h"

class Neuron : public dynamic_elem {
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

    Neuron(bool is_input, bool is_output);

    void fire(int time_step);

    float introduce_targets(float target, int timestep);

    float introduce_targets(float target, int timestep, float gamma, float lambda);

    float introduce_targets(float target, int timestep, float gamma, float lambda, bool no_grad);

    void propagate_error();

//    Returns the gradient of the post activation w.r.t pre-activation
    virtual float backward(float output_grad)  = 0;

    virtual float forward(float temp_value)  = 0;

    void propagate_deep_error();

    void update_utility();

    void mark_useless_weights();

    void prune_useless_weights();

    ~Neuron() = default;
};


class ReluNeuron : public Neuron{
 public:
    float backward(float output_grad);
    float forward(float temp_value);

    ReluNeuron(bool is_input, bool is_output);

};

class SigmoidNeuron : public Neuron{
 public:
    float backward(float output_grad);
    float forward(float temp_value);
    SigmoidNeuron(bool is_input, bool is_output);
};

class LinearNeuron : public Neuron{
 public:
    float backward(float output_grad);
    float forward(float temp_value);
    LinearNeuron(bool is_input, bool is_output);
};

class LeakyRelu : public Neuron{
    float negative_slope;
public:
    float backward(float output_grad);
    float forward(float temp_value);
    LeakyRelu(bool is_input, bool is_output, float negative_slope);
};

#endif  // INCLUDE_NN_NEURON_H_
