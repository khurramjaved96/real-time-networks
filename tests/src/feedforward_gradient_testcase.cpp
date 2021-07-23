//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include "../include/feedforward_gradient_testcase.h"

#include <iostream>
#include <vector>

#include "../include/fixed_feedforward_network.h"
#include "../../include/utils.h"
#include "../../include/environments/animal_learning/tracecondioning.h"


bool feedforwadtest() {
//
    TestCase my_network = TestCase(0.0, 5, 5);
    long long int time_step = 0;
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    float running_error = -1;

    std::vector<std::vector<float>> input_list;
    for (int a = 0; a < 300; a++) {
        std::vector<float> curr_inp;
        if (a < 100) {
            curr_inp.push_back(0);
            curr_inp.push_back(0);
            curr_inp.push_back(0);
        }
        else if (a < 200) {
            curr_inp.push_back((a - 100) * 0.01);
            curr_inp.push_back(10 - ((a - 100) * 0.1));
            curr_inp.push_back(a-100);
        } else {
            curr_inp.push_back(0);
            curr_inp.push_back(0);
            curr_inp.push_back(0);
        }
        input_list.push_back(curr_inp);
    }
    int counter = 0;
    float sum_of_activation=0;
    for (auto it : input_list) {
        my_network.set_input_values(it);
        my_network.step();
        std::vector<float> output = my_network.read_output_values();
        std::vector<float> output2 = my_network.read_all_values();
        sum_of_activation += output2[5];
//        std::cout << "counter = " << counter << std::endl;

//        print_vector(output);
        if (counter < 200 and counter >= 100) {
            output[0]++;
        }
        my_network.introduce_targets(output);
        counter++;
    }
//    std::cout << "Sum of activation "  << sum_of_activation << std::endl;
//    int counter_tt = 0;
//    for (auto it: my_network.all_synapses) {
//        std::cout << it->input_neuron->id + 1 << " " << it->output_neuron->id + 1  << " " << my_network.sum_of_gradients[counter_tt] << std::endl;
//        counter_tt++;
//    }
    std::vector<float> gt{-3.9099998474121094, 9.505999565124512, -45.89999771118164, 93.43999481201172, -391.0, 525.6000366210938, 4851.0, 587.7637329101562, 3002.000732421875, 315.5230712890625, 274.3168029785156, 611.5283203125};
    int counter_gt = 0;
    for(auto it: gt) {
        if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
            return false;
        }
        counter_gt++;
    }
    return true;

}

