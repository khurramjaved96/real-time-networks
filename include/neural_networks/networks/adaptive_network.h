//
// Created by Khurram Javed on 2021-04-01.
//

#ifndef FLEXIBLENN_ADAPTIVE_NETWORK_H
#define FLEXIBLENN_ADAPTIVE_NETWORK_H




#include "../synapse.h"
#include <vector>
#include "../neuron.h"
#include <vector>
#include <map>
#include <random>



class CustomNetwork{

    std::vector<neuron*> output_neuros;

    long long int time_step;
    std::vector<synapse*> all_synapses;
    std::vector<synapse*> output_synapses;
    std::vector<no_grad_synapse*> memories;
    std::mt19937 mt;
    std::vector<synapse*> memory_feature_weights;
public:
    std::vector<neuron*> all_neurons;
    std::vector<neuron*> input_neurons;
    std::vector<neuron*> new_features;
    #TODO CustomNetwork(float step_size, int width, int num_layers, int sparsity, int seed);
    CustomNetwork(float step_size, int width, int seed);
    ~CustomNetwork();
    void print_graph(neuron* root);
    void viz_graph();
    void set_print_bool();
    std::string get_viz_graph();
    long long int get_timestep();

    void set_input_values(std::vector<float> const &input_values);
    void step();
    std::vector<float> read_output_values();
    std::vector<float> read_all_values();
    float introduce_targets(std::vector<float> targets);
    #TODO float introduce_targets(std::vector<float> targets, std::vector<bool> no_grad);
    float introduce_targets(std::vector<float> targets, float gamma, float lambda);
    int get_input_size();
    int get_total_synapses();
    void add_memory(float step_size);
    void add_feature(float step_size);
    std::vector<float> get_memory_weights();
    void initialize_network(const std::vector<std::vector<float>>& input_batch);
};



#endif //FLEXIBLENN_ADAPTIVE_NETWORK_H
