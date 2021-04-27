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


int main(int argc, char *argv[]) {

//    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0";
    std::cout << "Program started \n";
    float gamma = 1 - 1.0/(3.0);
    TraceConditioning tc = TraceConditioning(std::pair<int, int>(20, 20), std::pair<int, int>(60, 100), 2, 2);
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

    Metric synapses_metric = Metric(my_experiment.database_name, "error_table", std::vector<std::string>{"step", "run", "error"}, std::vector<std::string>{"int", "int", "real"}, std::vector<std::string>{"step", "run"});
//    std::cout << "Database stuff done \n";
    CustomNetwork my_network = CustomNetwork(my_experiment.get_float_param("step_size"),
                                             my_experiment.get_int_param("width"), my_experiment.get_int_param("seed"));
//    my_network.set_print_bool();
    long long int time_step = 0;
//    for(auto &it : my_network.input_neurons)
//        my_network.print_graph(it);
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    auto start = std::chrono::steady_clock::now();
    float running_error = -1;
    std::vector<std::vector<std::string>> error_logger;
    for (int counter = 0; counter < my_experiment.get_int_param("steps"); counter++) {

        std::vector<std::string> error;


        std::vector<float> temp_target;
        if (counter % 3 == 0) {
            my_network.set_input_values(std::vector<float>{1, 1});
            temp_target.push_back(0.50);
            temp_target.push_back(-0.50);
        } else if (counter % 3 == 1) {
            my_network.set_input_values(std::vector<float>{1, 0});
            temp_target.push_back(-0.1);
            temp_target.push_back(-0.3);
        } else {
            my_network.set_input_values(std::vector<float>{0, 1});
            temp_target.push_back(-0.50);
            temp_target.push_back(0.40);
        }

        if (running_error == -1)
            running_error = my_network.introduce_targets(temp_target);
        else
            running_error = running_error * 0.99 + 0.01 * my_network.introduce_targets(temp_target);



        long long int time_step = 0;


//
        my_network.step();

        if(counter % 100 == 0)
        {
            error.push_back(std::to_string(counter));
            error.push_back(std::to_string(my_experiment.get_int_param("run")));

            error.push_back(std::to_string(running_error));
            error_logger.push_back(error);

        }
        if(counter % 10000 == 0)
        {
            synapses_metric.add_values(error_logger);
            error_logger.clear();
        }
        if (counter % 1000 == 0 || counter % 1000 == 999 || counter % 1000 == 998) {
            std::cout << "### STEP = " << counter << std::endl;
            std::cout << "Running error = " << running_error << std::endl;


            print_vector(my_network.read_output_values());
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
//
// Created by Khurram Javed on 2021-04-01.
//

