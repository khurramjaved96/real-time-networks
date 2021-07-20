//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>

#include "../include/neural_networks/networks/test.h"
#include "../include/utils.h"
#include "../include/animal_learning/tracecondioning.h"


int feedforwadtest() {
//
    TestCase my_network = TestCase(0.0, 5, 5);
    long long int time_step = 0;
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    float running_error = -1;

    std::vector<std::vector<float>> input_list;
    for (int a = 0; a < 200; a++) {
        std::vector<float> curr_inp;
        if (a < 100) {
            curr_inp.push_back(a * 0.01);
            curr_inp.push_back(10 - (a * 0.1));
            curr_inp.push_back(a);
        } else {
            curr_inp.push_back(0);
            curr_inp.push_back(0);
            curr_inp.push_back(0);
        }
        input_list.push_back(curr_inp);
    }
    int counter = 0;
    for (auto it : input_list) {
        my_network.set_input_values(it);
        my_network.step();
        std::vector<float> output = my_network.read_output_values();
        std::cout << "counter = " << counter << std::endl;

        print_vector(output);
        if (counter < 200) {
            output[0]++;
        }
        my_network.introduce_targets(output);

        counter++;
    }

    print_vector(my_network.sum_of_gradients);
    return 0;

}

