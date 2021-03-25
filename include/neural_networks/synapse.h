//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef BENCHMARKS_SYNAPSE_H
#define BENCHMARKS_SYNAPSE_H


#include <vector>
#include "neuron.h"




class synapse{
public:
    float weight;
    neuron *input_neurons;
    neuron *output_neurons;
    explicit synapse(neuron *input, neuron *output, float w);
//    void process_input();
    void step();
//    void sync();
};

#endif //BENCHMARKS_SYNAPSE_H