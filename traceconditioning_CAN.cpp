#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "include/utils.h"
#include <map>
#include <string>

#include "include/neural_networks/networks/adaptive_network.h"
#include "include/neural_networks/neural_network.h"
#include "include/experiment/Experiment.h"
#include "include/neural_networks/utils.h"
#include "include/experiment/Metric.h"
#include "include/animal_learning/tracecondioning.h"
#include "include/neural_networks/networks/adaptive_network.h"

//
int main(int argc, char *argv[]) {

//    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0 --step_size 0.0001";
    std::cout << "Program started \n";
    float gamma = 1 - 1.0 / (20.0);
    float lambda = 0.90;
    TraceConditioning tc1 = TraceConditioning(std::pair<int, int>(20, 20), std::pair<int, int>(20, 20),
                                              std::pair<int, int>(60, 100), 5, 2);
    TraceConditioning tc2 = TraceConditioning(std::pair<int, int>(4, 4), std::pair<int, int>(4, 4),
                                              std::pair<int, int>(60, 100), 5, 2);
    TraceConditioning tc = TraceConditioning(std::pair<int, int>(20, 20), std::pair<int, int>(20, 20),
                                             std::pair<int, int>(60, 100), 5, 2);
    for (int temp = 0; temp < 200; temp++) {
        std::vector<float> cur_state = tc.step();
        cur_state[2] = tc.get_target(gamma);
        print_vector(cur_state);
    }
//    exit(1);
    Experiment my_experiment = Experiment(argc, argv);
    std::cout << "Experiment object created \n";
    int width = my_experiment.get_int_param("width");

    Metric synapses_metric = Metric(my_experiment.database_name, "error_table",
                                    std::vector<std::string>{"step", "run", "error", "error_type"},
                                    std::vector<std::string>{"int", "int", "real", "int"},
                                    std::vector<std::string>{"step", "run", "error_type"});
    Metric graph_state_metric = Metric(my_experiment.database_name, "graph",
                                       std::vector<std::string>{"step", "run", "graph_data"},
                                       std::vector<std::string>{"int", "int", "MEDIUMTEXT"},
                                       std::vector<std::string>{"step", "run"});
    Metric state_metric = Metric(my_experiment.database_name, "state",
                                 std::vector<std::string>{"step", "run", "elem1", "elem2", "gt", "pred", "target"},
                                 std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real"},
                                 std::vector<std::string>{"step", "run"});
    Metric network_size_metric = Metric(my_experiment.database_name, "synapses",
                                        std::vector<std::string>{"step", "run", "total_synapses"},
                                        std::vector<std::string>{"int", "int", "int"},
                                        std::vector<std::string>{"step", "run"});
//    std::cout << "Database stuff done \n";
//    CustomNetwork my_network = CustomNetwork(my_experiment.get_float_param("step_size"),
//                                             my_experiment.get_int_param("width"), my_experiment.get_int_param("seed"));
    CustomNetwork my_network = CustomNetwork(my_experiment.get_float_param("step_size"),
                                             my_experiment.get_int_param("width"), my_experiment.get_int_param("seed"));


    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    my_network.viz_graph();
    auto start = std::chrono::steady_clock::now();
    std::vector<float> running_error;
    running_error.push_back(-1);
    running_error.push_back(-1);
    std::vector<std::vector<std::string>> error_logger;
    std::vector<std::vector<std::string>> state_logger;
    std::vector<std::vector<std::string>> network_size_logger;
    std::vector<std::vector<std::string>> graph_data_logger;


    float prediction = 0;
    float real_target = 0;
    float R = 0;
    float old_R = 0;


    for (int counter = 0; counter < my_experiment.get_int_param("steps"); counter++) {


        std::vector<float> temp_target;

        auto state_current = tc.step();
        std::vector<float> new_vec;
        for (auto &it: state_current) {
            new_vec.push_back(it);
        }

        my_network.set_input_values(state_current);


        old_R = R;
        R = tc.get_US();


        my_network.step();

        prediction = my_network.read_output_values()[0];

        real_target = tc.get_target(gamma);
//        real_target_long = tc.get_target_long(gamma);
        float target = prediction * gamma + old_R;
//        float target_long = prediction_long * gamma + old_R_long;
        if (counter > 0) {
            float error_short = (my_network.read_output_values()[0] - real_target) *
                                (my_network.read_output_values()[0] - real_target);
//            float error_long = (my_network.read_output_values()[1] - real_target_long) *
//                               (my_network.read_output_values()[1] - real_target_long);
            temp_target.push_back(target);
//            temp_target.push_back(target_long);
//            temp_target.push_back(0);
            my_network.introduce_targets(temp_target, gamma, lambda);
            if (running_error[0] == -1) {
                running_error[0] = error_short;
//                running_error[1] = error_long;
            } else {
                running_error[0] = running_error[0] * 0.999 + 0.001 * error_short;
//                running_error[1] = running_error[1] * 0.999 + 0.001 *error_long;
            }
        }
        if (counter % 300 == 0) {
            std::vector<std::string> error;
            error.push_back(std::to_string(counter));
            error.push_back(std::to_string(my_experiment.get_int_param("run")));
            error.push_back(std::to_string(running_error[0]));
            error.push_back(std::to_string(0));
            error_logger.push_back(error);

            std::vector<std::string> network_size;
            network_size.push_back(std::to_string(counter));
            network_size.push_back(std::to_string(my_experiment.get_int_param("run")));
            network_size.push_back(std::to_string(my_network.get_total_synapses()));
            network_size_logger.push_back(network_size);


        }
        if (counter % 50000 < 200) {
            std::vector<std::string> state_string;
            std::vector<float> cur_state = tc.get_state();
            state_string.push_back(std::to_string(counter));
            state_string.push_back(std::to_string(my_experiment.get_int_param("run")));
            state_string.push_back(std::to_string(cur_state[0]));
            state_string.push_back(std::to_string(cur_state[1]));
            state_string.push_back(std::to_string(real_target));
            state_string.push_back(std::to_string(my_network.read_output_values()[0]));
            state_string.push_back(std::to_string(target));
            state_logger.push_back(state_string);

            cur_state.push_back(real_target);
            cur_state.push_back(my_network.read_output_values()[0]);
            cur_state.push_back(target);
            print_vector(cur_state);
        }
        if (counter % 5000000 == 4999999) {
            tc = tc2;
        }
//        if (counter % 2000000 == 1999999) {
//            tc = tc1;
//        }
        if (counter % 80000 == 79999) {
//            std::vector<float> new_features;
//            std::cout << "NEW FEATURES \n\n";
//            for(auto it: my_network.new_features)
//                new_features.push_back(it->value);
//            print_vector(new_features);
            for (int a = 0; a < 1; a++)
                my_network.add_feature(my_experiment.get_float_param("step_size"));

            std::string g = my_network.get_viz_graph();
            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(counter));
            graph_data.push_back(std::to_string(my_experiment.get_int_param("run")));
            graph_data.push_back(g);
            graph_data_logger.push_back(graph_data);

        }

        if (counter % 100000 == 99998) {
            print_vector(my_network.get_memory_weights());
            std::cout << "Pushing results" << std::endl;
            synapses_metric.add_values(error_logger);
            std::cout << "Results added " << std::endl;
            std::cout << "Len = " << error_logger.size() << std::endl;
//            exit(1);
            error_logger.clear();

            network_size_metric.add_values(network_size_logger);
            network_size_logger.clear();

            state_metric.add_values(state_logger);
            state_logger.clear();

            graph_state_metric.add_values(graph_data_logger);
            graph_data_logger.clear();


        }
        if (counter % 10000 == 0 || counter % 10000 == 999 || counter % 10000 == 998) {
            std::cout << "### STEP = " << counter << std::endl;
            std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
            std::cout << "Running error = ";
            print_vector(running_error);

            my_network.set_print_bool();

//            for(auto &it : my_network.input_neurons)
//                my_network.print_graph(it);
//            for(auto &it : my_network.new_features)
//                my_network.print_graph(it);
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


    std::string g = my_network.get_viz_graph();
    std::vector<std::string> graph_data;
    graph_data.push_back(std::to_string(my_experiment.get_int_param("steps")));
    graph_data.push_back(std::to_string(my_experiment.get_int_param("run")));
    graph_data.push_back(g);
    graph_state_metric.add_value(graph_data);

    return 0;
}

//
// Created by Khurram Javed on 2021-04-01.
//

