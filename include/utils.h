//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef BENCHMARKS_UTILS_H
#define BENCHMARKS_UTILS_H

#endif //BENCHMARKS_UTILS_H
#include <string>
#include "neural_networks/neuron.h"
#include "neural_networks/synapse.h"

void print_vector(std::vector<int> const &v);
void print_vector(std::vector<float> const &v);
void print_matrix(std::vector<std::vector <int>> const &v);
void print_matrix(std::vector<std::vector <float>> const &v);
class NetworkVisualizer{
    std::string dot_string;
    std::vector<neuron *> all_neurons;
    std::vector<no_grad_synapse *> no_grad_synapses;

public:
    NetworkVisualizer(std::vector<neuron *> all_neurons);
    NetworkVisualizer(std::vector<neuron *> all_neurons, std::vector<no_grad_synapse*> no_grad_synapses);
    void generate_dot(int time_step);
    std::string get_graph(int time_step);
    std::string get_graph_detailed(int time_step);
    void generate_dot_detailed(int time_step);
};
