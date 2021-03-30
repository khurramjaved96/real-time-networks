#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "include/utils.h"
#include <map>
#include <string>

#include "include/neural_networks/neural_network.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"

float relu(float x) {
    if (x < 0) return 0;
    return x;
}

int main(int argc, char *argv[]) {


    std::map<std::string, std::string> t;
    Experiment my_experiment = Experiment(argc, argv);

    int width = 50;
    int total_layers = 30;

    Metric synapses_metric = Metric(my_experiment.database_name, "synapses_metric", std::vector<std::string>{"step", "run", "no"}, std::vector<std::string>{"int", "int", "int"}, std::vector<std::string>{"step", "run"});
    NeuralNetwork my_network = NeuralNetwork(total_layers, width);
    std::vector<float> input_vector;
    for (int i = 0; i < my_network.input_neurons.size(); i++) {
        input_vector.push_back(1);
    }

    auto start = std::chrono::steady_clock::now();
    int total_steps = 10;
    for (int steps = 0; steps < total_steps; steps++) {
        synapses_metric.add_value(std::vector<std::string> {std::to_string(steps), std::to_string(my_experiment.run), std::to_string(my_network.all_synapses.size()) });
        if (steps % 1000 == 0) {
            std::cout << "Step = " << steps << std::endl;
            std::cout << "Total synapses = " << my_network.all_synapses.size() << std::endl;
        }
//        print_vector(my_network.read_output_values());
        my_network.set_input_values(input_vector);
        my_network.step();
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / total_steps
              << " ms" << std::endl;
//    std::cout << "Time passed = " << time(NULL) - current_time << std::endl;
//    print_matrix(my_network.adjacency_matric);
    print_vector(my_network.read_output_values());
    my_network.update_depth_matrix();
    return 0;
}
//
