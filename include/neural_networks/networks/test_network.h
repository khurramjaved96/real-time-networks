//
// Created by Khurram Javed on 2021-04-01.
//

#ifndef FLEXIBLENN_TEST_NETWORK_H
#define FLEXIBLENN_TEST_NETWORK_H




#include "../synapse.h"
#include <vector>
#include "../neuron.h"
#include <vector>
#include <map>



class CustomNetwork{

    std::vector<neuron*> output_neuros;
    std::vector<neuron*> all_neurons;
    long long int time_step;
    std::vector<synapse*> all_synapses;
    std::vector<synapse*> output_synapses;
    std::vector<no_grad_synapse*> memories;
    std::vector<synapse*> memory_feature_weights;
public:
    std::vector<neuron*> input_neurons;
    CustomNetwork(float step_size, int width, int seed);
    ~CustomNetwork();
    void print_graph(neuron* root);
    void set_print_bool();

    void set_input_values(std::vector<float> const &input_values);
    void step();
    std::vector<float> read_output_values();
    std::vector<float> read_all_values();
    float introduce_targets(std::vector<float> targets);
    int get_input_size();
    int get_total_synapses();
    void add_memory(float step_size);
    std::vector<float> get_memory_weights();
};



#endif //FLEXIBLENN_TEST_NETWORK_H
