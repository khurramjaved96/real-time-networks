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
#include "../dynamic_elem.h"



class ContinuallyAdaptingNetwork{



    long long int time_step;

    std::mt19937 mt;


public:
    std::vector<neuron*> output_neurons;
    std::vector<synapse*> all_synapses;
    std::vector<synapse*> output_synapses;
    std::vector<dynamic_elem *> all_heap_elements;
//    std::vector<neuron *> all_heap_neurons;
//    std::vector<synapse *> all_heap_synapses;
    void college_garbage();
    std::vector<neuron*> all_neurons;
    std::vector<neuron*> input_neurons;
//    std::vector<neuron*> new_features;
    ContinuallyAdaptingNetwork(float step_size, int num_input, int num_output, int seed);
    ~ContinuallyAdaptingNetwork();
    void print_graph(neuron* root);
    void viz_graph();
    void set_print_bool();
    std::string get_viz_graph();
    long long int get_timestep();

    void set_input_values(std::vector<float> const &input_values);
    void reset_trace();
    void step();
    std::vector<float> read_output_values();
    std::vector<float> read_all_values();
    float introduce_targets(std::vector<float> targets);
    float introduce_targets(std::vector<float> targets, float gamma, float lambda);
    float introduce_targets(std::vector<float> targets, float gamma, float lambda, std::vector<bool> no_grad);
    int get_input_size();
    int get_total_synapses();
//    void add_memory(float step_size);
    void add_feature(float step_size);
};





#endif //FLEXIBLENN_ADAPTIVE_NETWORK_H
