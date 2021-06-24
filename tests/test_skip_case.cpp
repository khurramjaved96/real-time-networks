//
// Created by taodav on 23/6/21.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "../include/utils.h"
#include <map>
#include <string>

#include "../include/neural_networks/networks/test_skip.h"
#include "../include/animal_learning/tracecondioning.h"


int main(int argc, char *argv[]) {

//
    TestSkip my_network = TestSkip(0.0, 5, 5);
//    my_network.set_print_bool();
    long long int time_step = 0;
//    for(auto &it : my_network.is_input_neuron)
//        my_network.print_graph(it);
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    auto start = std::chrono::steady_clock::now();
    float running_error = -1;

    std::vector<std::vector<float>> input_list;
    for(int a = 0; a < 14; a++)
    {
        std::vector<float> curr_inp;
        if(a < 3) {
            curr_inp.push_back((a + 1));
        }
        else{
            curr_inp.push_back(0);
        }
        input_list.push_back(curr_inp);
    }
    int counter = 0;
    for(auto it:input_list)
    {
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
