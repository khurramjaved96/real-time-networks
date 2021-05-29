//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef BENCHMARKS_SYNAPSE_H
#define BENCHMARKS_SYNAPSE_H


#include <vector>
//#include "neuron.h"
#include "message.h"
#include <queue>
#include <utility>

class neuron;


class synapse {
public:
    long long int id;
    static long long int synapse_id;
    bool useless;
    int age;
    bool mark_delete;
    float weight;
    float credit;
    float trace;
    float credit_activation_idbd;
    float step_size;
    bool print_status;
    bool pass_gradients;
    float b1;
    float b2;
    float beta_step_size;
    float h_step_size;
    float idbd;
    bool log;
    std::queue<message> grad_queue;
    std::queue<message> grad_queue_weight_assignment;
    std::queue<std::pair<float, int>> weight_assignment_past_activations;
    neuron *input_neuron;
    neuron *output_neuron;

    explicit synapse(neuron *input, neuron *output, float w, float step_size);

    void read_gradients();

    void update_credit();

//    void process_input();
    void step();
    void block_gradients();

    void read_gradient();

    void zero_gradient();

    void update_weight();
    void turn_on_idbd();
    void assign_credit();

};

class no_grad_synapse{
public:
    neuron *input_neurons;
    neuron *output_neurons;

    explicit no_grad_synapse(neuron *input, neuron *output);

    void copy_activation(int time_step);
};

#endif //BENCHMARKS_SYNAPSE_H