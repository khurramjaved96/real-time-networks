//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef BENCHMARKS_NEURAL_NETWORK_H
#define BENCHMARKS_NEURAL_NETWORK_H


#include "synapse.h"
#include <vector>
#include "neuron.h"
#include <vector>
#include <map>
#include "dynamic_elem.h"



class NeuralNetwork
{
public:

    std::vector<neuron*> all_neurons;
    std::vector<neuron*> input_neurons;
    std::vector<neuron*> output_neurons;
    std::vector<synapse*> all_synapses;
    std::vector<std::vector<int>> adjacency_matric;
    NeuralNetwork(int total_layers, int width);
    void set_input_values(std::vector<float> const &input_values);
    void step();
    void update_depth_matrix();
    void delete_network();
    std::vector<float> read_output_values();
    void introduce_error(std::map<neuron, float> const &error_map);
};

#endif //BENCHMARKS_NEURAL_NETWORK_H