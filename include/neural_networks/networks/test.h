//
// Created by Khurram Javed on 2021-04-25.
//

#ifndef FLEXIBLENN_TEST_H
#define FLEXIBLENN_TEST_H


#include "../synapse.h"
#include <vector>
#include "../neuron.h"
#include <vector>
#include <map>



class TestCase{

    std::vector<neuron*> output_neuros;
    std::vector<neuron*> all_neurons;

    long long int time_step;
    std::vector<synapse*> all_synapses;
public:
    std::vector<float> sum_of_gradients;
    std::vector<neuron*> input_neurons;
    TestCase(float step_size, int width, int seed);
    ~TestCase();
    void print_graph(neuron* root);
    void set_print_bool();

    void set_input_values(std::vector<float> const &input_values);
    void step();
    std::vector<float> read_output_values();
    std::vector<float> read_all_values();
    std::vector<float> read_all_temp_values();
    float introduce_targets(std::vector<float> targets);
    int get_input_size();
    int get_total_synapses();
};



#endif //FLEXIBLENN_TEST_H
