//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "include/utils.h"
#include <map>
#include <string>

#include "include/neural_networks/networks/test.h"
#include "include/neural_networks/neural_network.h"
//#include "include/neural_networks/networks/adaptive_network.h"
#include "include/experiment/Experiment.h"
#include "include/neural_networks/utils.h"
#include "include/experiment/Metric.h"
#include "include/animal_learning/tracecondioning.h"


int main(int argc, char *argv[]) {

//
    TestCase my_network = TestCase(0.0, 5, 5);
//    my_network.set_print_bool();
    long long int time_step = 0;
//    for(auto &it : my_network.is_input_neuron)
//        my_network.print_graph(it);
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
//    for(int counter = 0; counter < 3; counter++)
//    {
//        std::vector<float> curr_inp;
//        curr_inp.push_back(0);
//        curr_inp.push_back(0);
//        curr_inp.push_back(0);
//        my_network.set_input_values(curr_inp);
//        std::vector<float> output = my_network.read_output_values();
//        output[0]++;
//        my_network.introduce_targets(output);
//        my_network.step();
//    }
    print_vector(my_network.sum_of_gradients);
    exit(1);
//    list_of_state_values = []
//    for a in range(0, 100):
//    list_of_state_values.append([a*0.01, 10 - (a*0.1), a])
//    std::vector<std::vector<std::string>> error_logger;
//    for (int counter = 0; counter < 1000; counter++) {
//
//        std::vector<std::string> error;
//
//
//        std::vector<float> temp_target;
//
////        temp_target.push_back(tc.get_target(gamma));
////        my_network.set_input_values(tc.step());
//
////        temp_target.push_back(0);
////        if (counter % 3 == 0) {
////            my_network.set_input_values(std::vector<float>{1, 1});
////            temp_target.push_back(0.50);
////            temp_target.push_back(-0.50);
////        } else if (counter % 3 == 1) {
////            my_network.set_input_values(std::vector<float>{1, 0});
////            temp_target.push_back(-0.1);
////            temp_target.push_back(-0.3);
////        } else {
////            my_network.set_input_values(std::vector<float>{0, 1});
////            temp_target.push_back(-0.50);
////            temp_target.push_back(0.40);
////        }
//
//        if (running_error == -1)
//            running_error = my_network.introduce_targets(temp_target);
//        else
//            running_error = running_error * 0.999 + 0.001 * my_network.introduce_targets(temp_target);
//
//
//
//        long long int time_step = 0;
//
//
////
//        my_network.step();
//
//        if(counter % 500 == 0)
//        {
//            error.push_back(std::to_string(counter));
//            error.push_back(std::to_string(my_experiment.get_int_param("run")));
//
//            error.push_back(std::to_string(running_error));
//            error_logger.push_back(error);
//
//        }
//        if(counter % 10000 < 100)
//        {
//            std::vector<float> cur_state = tc.get_state();
//            cur_state.push_back(tc.get_target(gamma) );
//            cur_state.push_back(my_network.read_output_values()[0]);
////            cur_state[2] = tc.get_target(gamma);
////            cur_state[3] = my_network.read_output_values()[0];
//            print_vector(cur_state);
//        }
//        if(counter % 10000 == 0)
//        {
//            synapses_metric.add_values(error_logger);
//            error_logger.clear();
//        }
//        if (counter % 1000 == 0 || counter % 1000 == 999 || counter % 1000 == 998) {
//            std::cout << "### STEP = " << counter << std::endl;
//            std::cout << "Running error = " << running_error << std::endl;
//
//
//            print_vector(my_network.read_output_values());
//            std::cout << "Target = " << tc.get_target(gamma) << std::endl;
//        }
//    }
//
//    auto end = std::chrono::steady_clock::now();
//    std::cout << "Elapsed time in milliseconds for per steps: "
//              << 1000000 / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
//                            my_experiment.get_int_param("steps"))
//              << " fps" << std::endl;
    return 0;
}

//
// Created by Khurram Javed on 2021-04-01.
//

