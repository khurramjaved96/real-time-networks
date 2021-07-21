//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <string>
#include <vector>
#include "neural_networks/neuron.h"

void print_vector(std::vector<int> const &v);

void print_vector(std::vector<float> const &v);

void print_matrix(std::vector<std::vector<int>> const &v);

void print_matrix(std::vector<std::vector<float>> const &v);

class NetworkVisualizer {
    std::string dot_string;
    std::vector<neuron *> all_neurons;

 public:
    explicit NetworkVisualizer(std::vector<neuron *> all_neurons);

    void generate_dot(int time_step);

    std::string get_graph(int time_step);

    std::string get_graph_detailed(int time_step);

    void generate_dot_detailed(int time_step);
};

#endif  // INCLUDE_UTILS_H_

