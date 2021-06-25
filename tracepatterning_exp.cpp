#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

//
// Created by Khurram Javed on 2021-04-01.
//



#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "include/utils.h"
#include <map>
#include <string>
#include <math.h>

#include "include/neural_networks/networks/adaptive_network.h"
#include "include/neural_networks/neural_network.h"
#include "include/experiment/Experiment.h"
#include "include/neural_networks/utils.h"
#include "include/experiment/Metric.h"
#include "include/animal_learning/tracecondioning.h"
#include "include/neural_networks/networks/adaptive_network.h"

/**
 * Our main entry function for running all experiments.
 * @param argc Number of arguments
 * @param argv This needs to include the following parameters:
 *  --run (int, 0), the run number
 *  --ISI_low (int, 14), the ISI is sampled based on a uniform distribution. What is the lower bound for this distribution?
 *  --ISI_high (int, 26), what is the upper bound for this distribution?
 *  --lambda (float, 0.0), parameter for an eligibility trace. What is our trace parameter?
 *  --seed (int, 2021), what is the seed we use?
 *  --width (int, 6), [NOT CURRENTLY USED] what is the width of our neural network?
 *  --step_size (float, 0.0001), step size parameter.
 *  --steps (int, 5000000), total number of steps to take in the experiment.
 * @return void
 */
int main(int argc, char *argv[]) {

//    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0 --step_size 0.0001";

    // Initialize everything
    Experiment my_experiment = Experiment(argc, argv);
    std::cout << "Program started \n";
    int interval = my_experiment.get_int_param("ISI_low");
    int interval_up = my_experiment.get_int_param("ISI_high");
    float gamma = 1.0 - 1.0 / double(interval_up);
    float lambda = my_experiment.get_float_param("lambda");

    // Initialize our dataset
    TracePatterning tc = TracePatterning(std::pair<int, int>(interval, interval_up),
                                          std::pair<int, int>(interval, interval_up),
                                          std::pair<int, int>(80, 120), 0, my_experiment.get_int_param("seed"));

    for (int temp = 0; temp < 200; temp++) {
        std::vector<float> cur_state = tc.step();
        cur_state[2] = tc.get_target(gamma);
        print_vector(cur_state);
    }

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
                                 std::vector<std::string>{"step", "run", "elem1", "elem2", "elem3", "elem4", "elem5",
                                                          "elem6", "gt", "pred", "target"},
                                 std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "real",
                                                          "real", "real", "real"},
                                 std::vector<std::string>{"step", "run"});
    Metric network_size_metric = Metric(my_experiment.database_name, "synapses",
                                        std::vector<std::string>{"step", "run", "total_synapses"},
                                        std::vector<std::string>{"int", "int", "int"},
                                        std::vector<std::string>{"step", "run"});

    // Initialize our network
    ContinuallyAdaptingNetwork my_network = ContinuallyAdaptingNetwork(my_experiment.get_float_param("step_size"),
                                                                       my_experiment.get_int_param("width"),
                                                                       my_experiment.get_int_param("seed"));


    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
//    my_network.viz_graph();
    auto start = std::chrono::steady_clock::now();
    std::vector<float> running_error;
    running_error.push_back(0);
    running_error.push_back(0);

    std::vector<std::vector<std::string>> error_logger;
    std::vector<std::vector<std::string>> state_logger;
    std::vector<std::vector<std::string>> network_size_logger;
    std::vector<std::vector<std::string>> graph_data_logger;

    float prediction = 0;
    float real_target = 0;
    float R = 0;
    float old_R = 0;

//  start taking steps!
    for (int counter = 0; counter < my_experiment.get_int_param("steps"); counter++) {

        std::vector<float> temp_target;

//      Get our current state
        auto state_current = tc.step();
        std::vector<float> new_vec;
        for (auto &it: state_current) {
            new_vec.push_back(it);
        }

//      Set our input into our NN
        my_network.set_input_values(state_current);

//      "reward" in this case is our unconditioned stimulus
        old_R = R;
        R = tc.get_US();

//      THIS is the main call for this network - fire our neurons a step, calculate gradients, backprop, update and
//      prune if necessary.
        my_network.step();

//      Get the predictions from our output neurons
        prediction = my_network.read_output_values()[0];

//      Now we calculate our bootstrapped TD target
        real_target = tc.get_target(gamma);
        float target = prediction * gamma + old_R;
        if (counter > 0) {
            float error_short = (my_network.read_output_values()[0] - real_target) *
                                (my_network.read_output_values()[0] - real_target);

            temp_target.push_back(target);

//          Here we put our targets into our output neurons and calculate our TD error.
            my_network.introduce_targets(temp_target, gamma, lambda);

            float beta = 0.9999;
            running_error[0] = running_error[0] * beta + (1 - beta) * error_short;
            running_error[1] = running_error[0] / (1 - pow(beta, counter));

        }

//      For logging purposes
        if (counter % 300 == 0) {
            std::vector<std::string> error;
            error.push_back(std::to_string(counter));
            error.push_back(std::to_string(my_experiment.get_int_param("run")));
            error.push_back(std::to_string(running_error[1]));
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
            for (int temp = 0; temp < 6; temp++)
                state_string.push_back(std::to_string(cur_state[temp]));
//            state_string.push_back(std::to_string(cur_state[1]));
            state_string.push_back(std::to_string(real_target));
            state_string.push_back(std::to_string(my_network.read_output_values()[0]));
            state_string.push_back(std::to_string(target));
            state_logger.push_back(state_string);

            cur_state.push_back(real_target);
            cur_state.push_back(my_network.read_output_values()[0]);
            cur_state.push_back(target);
            print_vector(cur_state);
        }

//      Generating new features every 80000 steps
        if (counter % 80000 == 79999) {
//          First remove all references to useless nodes and neurons
            my_network.collect_garbage();

//          Add 20 new features
            for (int a = 0; a < 20; a++)
                my_network.add_feature(my_experiment.get_float_param("step_size"));

        }

//      visualizations
        if (counter % 1000000 == 999999) {
            std::string g = my_network.get_viz_graph();
            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(counter));
            graph_data.push_back(std::to_string(my_experiment.get_int_param("run")));
            graph_data.push_back(g);
            graph_data_logger.push_back(graph_data);
        }

        if (counter % 100000 == 99998) {
//            print_vector(my_network.get_memory_weights());
            std::cout << "Pushing results" << std::endl;
            synapses_metric.add_values(error_logger);
            std::cout << "Results added " << std::endl;
            std::cout << "Len = " << error_logger.size() << std::endl;
//            exit(1);
            error_logger.clear();
            my_network.collect_garbage();

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
            std::cout << "Total elements = " << my_network.all_heap_elements.size() << std::endl;
            std::cout << "Output synapses = " << my_network.output_synapses.size() << std::endl;
            std::cout << "Total synapses = " << my_network.all_synapses.size() << std::endl;
            std::cout << "Output Neurons = " << my_network.output_neurons.size() << "\t"
                      << my_network.input_neurons.size() << std::endl;
            std::cout << "Total Neurons = " << my_network.all_neurons.size() << std::endl;
            my_network.set_print_bool();

        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << 1000000 / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                            my_experiment.get_int_param("steps"))
              << " fps" << std::endl;


    return 0;
}

