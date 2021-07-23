//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT


#include <vector>
#include "../../include/utils.h"

#include "../include/fixed_recurrent_network.h"
#include "../../include/environments/animal_learning/tracecondioning.h"


bool recurrent_network_test() {
    ContinuallyAdaptingRecurrentNetworkTest my_network = ContinuallyAdaptingRecurrentNetworkTest(0.0, 5, 5);
    std::vector<float> sum_of_gradients;
    for (auto it: my_network.all_synapses)
        sum_of_gradients.push_back(0);

    std::vector<std::vector<float>> input_list;

    for (int a = 0; a < 100; a++) {
        std::vector<float> curr_inp;
        curr_inp.push_back(0);
        curr_inp.push_back(0);
        input_list.push_back(curr_inp);
    }
    for (int a = 0; a < 100; a++) {
        std::vector<float> curr_inp;

        curr_inp.push_back(10 - (a * 0.1));
        curr_inp.push_back(a * 0.01);
        input_list.push_back(curr_inp);
    }
    for (int a = 0; a < 100; a++) {
        std::vector<float> curr_inp;
        curr_inp.push_back(0);
        curr_inp.push_back(0);
        input_list.push_back(curr_inp);
    }


    float sum_of_state = 0;
    int counter = 0;
    for (auto it:input_list) {
        my_network.set_input_values(it);
        my_network.step();
        std::vector<float> output = my_network.read_output_values();

        if (counter >= 100 and counter < 200) {
            sum_of_state += my_network.read_all_values()[2];
            output[0]++;
        }
        my_network.introduce_targets(output, 0, 0);
        int counter_temp = 0;
        for (auto it: my_network.all_synapses) {
            sum_of_gradients[counter_temp] += it->credit;
            counter_temp++;
        }
        counter++;
    }
// Ground truth is computed by running "recurrent_test.py" in python_scripts
    std::vector<float> gt{877.8438, 74.0906, 1824.9984, 1043.9813};
    for (int i = 0; i < sum_of_gradients.size(); i++) {
        if (std::abs(sum_of_gradients[i] - gt[i]) > 1e-3)
            return false;
    }
    return true;
}


//
// Created by Khurram Javed on 2021-04-01.
//

