//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include "../include/feedforward_gradient_testcase.h"

#include <iostream>
#include <vector>

#include "../include/fixed_feedforward_network.h"
#include "../../include/utils.h"
#include "../../include/animal_learning/tracecondioning.h"


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
        } else if (a < 200) {
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
        print_vector(it);
        my_network.set_input_values(it);
        my_network.step();
        std::vector<float> output = my_network.read_output_values();
        print_vector(output);
        std::vector<float> output2 = my_network.read_all_values();
        sum_of_activation += output2[6];
//        std::cout << "counter = " << counter << std::endl;

//        print_vector(output);
        if (counter < 200 and counter >= 100) {
            output[0]++;
        }
        my_network.introduce_targets(output);
        int counter_tt = 0;
        for (auto it: my_network.all_synapses) {
//            std::cout << it->credit << std::endl;
//            if(it->credit > 0)
//                std::cout << "Non zero credit\n";
            my_network.sum_of_gradients[counter_tt] += it->credit;
            counter_tt++;
        }
        counter++;
    }
    std::cout << "Sum of activation "  << sum_of_activation << std::endl;
    print_vector(my_network.sum_of_gradients);
    return true;

}

