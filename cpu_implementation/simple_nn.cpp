#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <chrono>
#include "include/utils.h"
#include <random>
#include <new>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "include/neural_networks/neuron.h"
#include "include/neural_networks/synapse.h"
#include "include/neural_networks/neural_network.h"


float relu(float x) {
    if (x < 0) return 0;
    return x;
}


extern "C" {
int NNTest(int width, int total_layers)
{
    NeuralNetwork my_network = NeuralNetwork(total_layers, width);
    std::vector<float> input_vector;
    for(int i = 0; i < my_network.input_neurons.size(); i++)
    {
        input_vector.push_back(1);
    }

    auto start = std::chrono::steady_clock::now();
    int total_steps = 10;
    for(int steps = 0; steps < total_steps; steps++)
    {
        if(steps%1000 == 0) {
            std::cout << "Step = " << steps << std::endl;
            std::cout << "Total synapses = " << my_network.all_synapses.size() << std::endl;
        }

        my_network.set_input_values(input_vector);
        my_network.step();
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/total_steps
              << " ms" << std::endl;
    print_vector(my_network.read_output_values());
    my_network.update_depth_matrix();
    return 0;
}
}

//
//int main(void) {
//
//    int width = 150;
//    int total_layers = 120;
//
////
//    NeuralNetwork my_network = NeuralNetwork(total_layers, width);
//    std::vector<float> input_vector;
//    for(int i = 0; i < my_network.input_neurons.size(); i++)
//    {
//        input_vector.push_back(1);
//    }
//
//    auto start = std::chrono::steady_clock::now();
//    int total_steps = 10;
//    for(int steps = 0; steps < total_steps; steps++)
//    {
//        if(steps%1000 == 0) {
//            std::cout << "Step = " << steps << std::endl;
//            std::cout << "Total synapses = " << my_network.all_synapses.size() << std::endl;
//        }
////        print_vector(my_network.read_output_values());
//        my_network.set_input_values(input_vector);
//        my_network.step();
//    }
//
//    auto end = std::chrono::steady_clock::now();
//    std::cout << "Elapsed time in milliseconds for per steps: "
//         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/total_steps
//         << " ms" << std::endl;
////    std::cout << "Time passed = " << time(NULL) - current_time << std::endl;
////    print_matrix(my_network.adjacency_matric);
//    print_vector(my_network.read_output_values());
//    my_network.update_depth_matrix();
//    return 0;
//}
////