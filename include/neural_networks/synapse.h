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
    float weight;
    float credit;
    float step_size;
    bool print_status;
    float b1;
    float b2;
    std::queue<message> grad_queue;
    neuron *input_neurons;
    neuron *output_neurons;

    explicit synapse(neuron *input, neuron *output, float w, float step_size);

    void read_gradients();

    void update_credit();

//    void process_input();
    void step();

    void read_gradient();

    void zero_gradient();

    void update_weight();

};

class no_grad_synapse{
public:
    neuron *input_neurons;
    neuron *output_neurons;

    explicit no_grad_synapse(neuron *input, neuron *output);

    void copy_activation();
};

#endif //BENCHMARKS_SYNAPSE_H