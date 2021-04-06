#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "include/utils.h"
#include <map>
#include <string>

#include "include/neural_networks/networks/test_network.h"
#include "include/neural_networks/neural_network.h"
#include "include/experiment/Experiment.h"
#include "include/neural_networks/utils.h"
#include "include/experiment/Metric.h"


int main(int argc, char *argv[]) {

    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0";
    std::cout << "Program started \n";
    Experiment my_experiment = Experiment(argc, argv);
    std::cout << "Experiment object created \n";
    int width = my_experiment.get_int_param("width");

//    Metric synapses_metric = Metric(my_experiment.database_name, "total_synapses", std::vector<std::string>{"step", "run", "no"}, std::vector<std::string>{"int", "int", "int"}, std::vector<std::string>{"step", "run"});
//    std::cout << "Database stuff done \n";
    CustomNetwork my_network = CustomNetwork();
    my_network.set_print_bool();
    long long int time_step = 0;
    for(auto &it : my_network.input_neurons)
        my_network.print_graph(it);
    for(int counter = 0; counter<my_experiment.get_int_param("steps"); counter++){
        std::cout << "### STEP = " << counter << std::endl;

        if(counter < 5)
            my_network.set_input_values(std::vector<float>{1, 1});



        if(counter == 4){
//            print_vector(my_network.read_output_values());
            std::vector<float>  temp_target;
            temp_target.push_back(12);
//            print_vector(temp_target);
            my_network.introduce_targets(temp_target);

        }
        else{
            my_network.introduce_targets(my_network.read_output_values());
        }


        long long int time_step = 0;


//
        my_network.step();
        my_network.set_print_bool();
        for(auto &it : my_network.input_neurons)
            my_network.print_graph(it);
//        print_vector(my_network.read_all_values());
    }


//    std::vector<float> input_vector;
//    std::cout << "Network created " << std::endl;
//    for (int i = 0; i < my_network.get_input_size(); i++) {
//        input_vector.push_back(1);
//
//    auto un_random = uniform_random(my_experiment.get_int_param("seed"));
//    long int total_steps = my_experiment.get_int_param("steps");
//    std::vector<std::vector<std::string>> my_results;
//    auto start = std::chrono::steady_clock::now();
//    for (int steps = 0; steps < total_steps; steps++) {
//        auto inp = un_random.get_random_vector(width);
//        std::cout << "Step = " << steps << std::endl;
//        print_vector(inp);
////        std::cout << "Step = " << steps << std::endl;
////        my_results.push_back(std::vector<std::string> {std::to_string(steps), std::to_string(my_experiment.run), std::to_string(my_network.get_total_synapses()) });
//        if (steps % 1000 == 0) {
//            std::cout << "Step = " << steps << std::endl;
//            std::cout << "Total synapses = " << my_network.get_total_synapses() << std::endl;
//        }
//        my_network.set_input_values(inp);
//        my_network.introduce_targets(std::vector<float>(10));
//        my_network.step();
//        print_vector(my_network.read_output_values());
//    }
////    synapses_metric.add_values(my_results);
//
//    auto end = std::chrono::steady_clock::now();
//    std::cout << "Elapsed time in milliseconds for per steps: "
//              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / total_steps
//              << " ms" << std::endl;
////    std::cout << "Time passed = " << time(NULL) - current_time << std::endl;
////    print_matrix(my_network.adjacency_matric);
//    print_vector(my_network.read_output_values());
    return 0;
}
//
//
// Created by Khurram Javed on 2021-04-01.
//

