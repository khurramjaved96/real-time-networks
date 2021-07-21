//
// Created by haseebs on 4/29/21.
//

#ifndef INCLUDE_NEURAL_NETWORKS_NETWORKS_SIMPLE_NETWORK_H_
#define INCLUDE_NEURAL_NETWORKS_NETWORKS_SIMPLE_NETWORK_H_


#include <vector>
#include <map>
#include "../synapse.h"
#include "../neuron.h"

class SimpleNetwork {
    std::vector<neuron *> output_neurons;
    std::vector<neuron *> all_neurons;
    int64_t time_step;
    std::vector<synapse *> all_synapses;

 public:
    std::vector<neuron *> input_neurons;

    SimpleNetwork(float step_size, int width, int seed);

    ~SimpleNetwork();

    void print_graph(neuron *root);

    void set_print_bool();

    void set_input_values(std::vector<float> const &input_values);

    void step();

    void initialize_network(std::vector<std::vector<float>> const &input_batch);

    std::vector<float> read_output_values();

    std::vector<float> read_all_values();

    float introduce_targets(std::vector<float> targets);

    int get_input_size();

    int get_total_synapses();
};

#endif  // INCLUDE_NEURAL_NETWORKS_NETWORKS_SIMPLE_NETWORK_H_
