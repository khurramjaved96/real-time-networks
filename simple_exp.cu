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
#include "src/hybrid_code/queue.cu"
#include "include/animal_learning/tracecondioning.h"
#include "include/neural_networks/networks/simple_network.h"


int main(int argc, char *argv[]) {

//    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0 --step_size 0.0001";
    std::cout << "Program started \n";
    float gamma = 1 - 1.0/(3.0);
    TraceConditioning tc = TraceConditioning(std::pair<int, int>(4, 4), std::pair<int, int>(15, 15), std::pair<int, int>(60, 100), 5, 2);
    for(int temp = 0; temp<200; temp++)
    {
        std::vector<float> cur_state = tc.step();
        cur_state[2] = tc.get_target(gamma);
        print_vector(cur_state);
    }
//    exit(1);
    Experiment my_experiment = Experiment(argc, argv);
    std::cout << "Experiment object created \n";
    int width = my_experiment.get_int_param("width");

    Metric synapses_metric = Metric(my_experiment.database_name, "error_table", std::vector<std::string>{"step", "run", "error", "error_type"}, std::vector<std::string>{"int", "int", "real", "int"}, std::vector<std::string>{"step", "run", "error_type"});
//    std::cout << "Database stuff done \n";
//    CustomNetwork my_network = CustomNetwork(my_experiment.get_float_param("step_size"),
//                                             my_experiment.get_int_param("width"), my_experiment.get_int_param("seed"));
    SimpleNetwork my_network = SimpleNetwork(my_experiment.get_float_param("step_size"),
                                             my_experiment.get_int_param("width"), my_experiment.get_int_param("seed"));

    //get a sequence of data for data-driven initialization
    std::vector<std::vector<float>> input_batch;
    input_batch.reserve(500);
    for(int temp=0; temp<500; temp++)
        input_batch.push_back(tc.step());
    my_network.initialize_network(input_batch);

//    my_network.set_print_bool();

//    for(auto &it : my_network.input_neurons)
//        my_network.print_graph(it);
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    auto start = std::chrono::steady_clock::now();
    std::vector<float> running_error;
    running_error.push_back(-1);
    running_error.push_back(-1);
    std::vector<std::vector<std::string>> error_logger;
    float prediction = 0;

    float prediction_long = 0;

    float real_target = 0;
    float real_target_long = 0;
    float R = 0;
    float  old_R = 0;
    float R_long = 0;
    float old_R_long = 0;
    for (int counter = 0; counter < my_experiment.get_int_param("steps"); counter++) {



        std::vector<std::string> error;
        std::vector<float> temp_target;

        auto state_current = tc.step();
        std::vector<float> new_vec;
        for(auto & it: state_current){
            new_vec.push_back(it);
        }
//        new_vec[1] = 0;
//        new_vec[2] = 0;
        my_network.set_input_values(state_current);


        old_R = R;
        R = tc.get_US();

        old_R_long = R_long;
        R_long = tc.get_long_US();
        my_network.step();

        prediction = my_network.read_output_values()[0];

        prediction_long = my_network.read_output_values()[1];

        real_target = tc.get_target(gamma);
        real_target_long = tc.get_target_long(gamma);
        float target = prediction * gamma + old_R;
        float target_long = prediction_long * gamma + old_R_long;
        if(counter > 0) {
            float error_short = (my_network.read_output_values()[0] - real_target) *
                          (my_network.read_output_values()[0] - real_target);
            float error_long = (my_network.read_output_values()[1] - real_target_long) *
                               (my_network.read_output_values()[1] - real_target_long);
            temp_target.push_back(target);
            temp_target.push_back(target_long);
            my_network.introduce_targets(temp_target);
            if (running_error[0] == -1) {
                running_error[0] = error_short;
                running_error[1] = error_long;
            }

            else {
                running_error[0] = running_error[0] * 0.999 + 0.001 *error_short;
                running_error[1] = running_error[1] * 0.999 + 0.001 *error_long;
            }
        }
        if(counter % 300 == 0)
        {
            error.push_back(std::to_string(counter));
            error.push_back(std::to_string(my_experiment.get_int_param("run")));
            error.push_back(std::to_string(running_error[0]));
            error.push_back(std::to_string(0));

            error_logger.push_back(error);
            error.clear();
            error.push_back(std::to_string(counter));
            error.push_back(std::to_string(my_experiment.get_int_param("run")));
            error.push_back(std::to_string(running_error[1]));
            error.push_back(std::to_string(1));

            error_logger.push_back(error);

        }
        if(counter % 10000 < 500)
        {
            std::vector<float> cur_state = tc.get_state();
            cur_state.push_back(real_target);
            cur_state.push_back(my_network.read_output_values()[0]);
            cur_state.push_back(target);
            cur_state.push_back(real_target_long);
            cur_state.push_back(my_network.read_output_values()[1]);
            cur_state.push_back(target_long);
//            cur_state[2] = tc.get_target(gamma);
//            cur_state[3] = my_network.read_output_values()[0];
            print_vector(cur_state);
        }
        if(counter%500000  == 499999)
        {
            my_network.add_memory(my_experiment.get_float_param("step_size"));
            my_network.add_memory(my_experiment.get_float_param("step_size"));
            my_network.add_memory(my_experiment.get_float_param("step_size"));


        }
//
        if(counter % 10000 == 9998)
        {
            print_vector(my_network.get_memory_weights());
            std::cout << "Pushing results" << std::endl;
            synapses_metric.add_values(error_logger);
            std::cout << "Results added " << std::endl;
            std::cout << "Len = " << error_logger.size() << std::endl;
//            exit(1);
            error_logger.clear();
        }
        if (counter % 10000 == 0 || counter % 10000 == 999 || counter % 10000 == 998) {
            std::cout << "### STEP = " << counter << std::endl;
            std::cout << "Running error = ";
            print_vector(running_error);
//            my_network.print_memory_feature_weights();

//            print_vector(my_network.read_output_values());
//            std::cout << "Target = " << tc.get_target(gamma) << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << 1000000 / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                            my_experiment.get_int_param("steps"))
              << " fps" << std::endl;
    return 0;
}

//
// Created by Khurram Javed on 2021-04-01.
//

