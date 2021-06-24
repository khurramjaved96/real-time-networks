//
// Created by taodav on 24/6/21.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "../include/utils.h"

#include "../include/neural_networks/networks/test_add_delete.h"
#include "../include/animal_learning/tracecondioning.h"


int main(int argc, char *argv[]) {
    std::string test_type = "add_neuron";
    if (argc > 1) {
        test_type = argv[1];
    }

    TestAddDelete my_network = TestAddDelete(0.0, 5, 5);
    long long int time_step = 0;
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    auto start = std::chrono::steady_clock::now();
    float running_error = -1;

    std::vector<std::vector<float>> input_list;
    for(int a = 0; a < 200; a++)
    {
        std::vector<float> curr_inp;
        if(a < 100) {
            curr_inp.push_back(a * 0.01);
            curr_inp.push_back(10 - (a * 0.1));
            curr_inp.push_back(a);
        }
        else{
            curr_inp.push_back(0);
            curr_inp.push_back(0);
            curr_inp.push_back(0);
        }
        input_list.push_back(curr_inp);
//        print_vector(curr_inp);
    }
    int counter = 0;
    for(auto it:input_list)
    {
        if (counter == 24) {
            if (test_type == "add_neuron") {
                my_network.add_feature(0.0);
            } else if (test_type == "delete_neuron") {
                my_network.delete_feature();
            }
        }
        my_network.set_input_values(it);
        my_network.step();
        std::vector<float> output = my_network.read_output_values();
        std::cout << "counter = " << counter << std::endl;
        print_vector(output);

//        std::cout << "All neuron output values" << std::endl;
//        print_vector(my_network.read_all_temp_values());

        if(counter < 200) {
            output[0]++;
        }
        my_network.introduce_targets(output);
        print_vector(my_network.sum_of_gradients);

        counter ++;
    }
    print_vector(my_network.sum_of_gradients);
    exit(1);
}

//
// Created by Khurram Javed on 2021-04-01.
//

