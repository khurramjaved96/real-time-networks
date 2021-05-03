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
#include "include/environments/copy_task.h"
#include "include/animal_learning/tracecondioning.h"
#include "include/neural_networks/networks/test_network.h"


int main(int argc, char *argv[]) {
//    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0 --step_size 0.0001";
//    __builtin_trap();
    std::cout << "Program started \n";
    Experiment my_experiment = Experiment(argc, argv);
    std::cout << "Experiment object created \n";
    int width = my_experiment.get_int_param("width");

    Metric synapses_metric = Metric(my_experiment.database_name, "error_table", std::vector<std::string>{"step", "datatime", "seq_length", "run", "error"}, std::vector<std::string>{"int", "int", "int", "int", "real"}, std::vector<std::string>{"step",  "run" });
    Metric graph_state = Metric(my_experiment.database_name, "graph", std::vector<std::string>{"step", "run", "graph_data"}, std::vector<std::string>{"int", "int", "MEDIUMTEXT"}, std::vector<std::string>{"step", "run"});
    CustomNetwork my_network = CustomNetwork(my_experiment.get_float_param("step_size"),
                                             my_experiment.get_int_param("width"), my_experiment.get_int_param("seed"));

//    //get a sequence of data for data-driven initialization
//    std::vector<std::vector<float>> input_batch;
//    input_batch.reserve(500);
//    for(int temp=0; temp<500; temp++)
//        input_batch.push_back(tc.step());
//    my_network.initialize_network(input_batch);

//    my_network.set_print_bool();
//
//    for(auto &it : my_network.input_neurons)
//        my_network.print_graph(it);
    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    my_network.viz_graph();
    auto start = std::chrono::steady_clock::now();

    CopyTask env = CopyTask(my_experiment.get_int_param("seed"));
    float running_error = -1;
    float target = 0;
    float prediction = 0;
    float last_err = 1;
    int current_seq_length = 1;
    std::vector<std::vector<std::string>> error_logger;

    std::cout << "Flag Bit \t Pred Bit \t Target \t Pred \t Seq_len \t Datatime" << std::endl;
    for (int counter = 0; counter < my_experiment.get_int_param("steps"); counter++) {

        auto state_current = env.step(last_err);
        my_network.set_input_values(state_current);
        my_network.step();
        prediction = my_network.read_output_values()[0];

        target = env.get_target();
        my_network.introduce_targets(std::vector<float>{target});

//        float error = (prediction - target) * (prediction - target);
        float error = prediction - target;
        last_err = error;
        if (running_error == -1)
            running_error = error;
        else
            running_error = running_error * 0.999 + 0.001 *error;

        if(env.get_L() > current_seq_length || counter % 300 == 0)
        {
            std::vector<std::string> error_vec;
            error_vec.push_back(std::to_string(counter));
            error_vec.push_back(std::to_string(env.get_data_timestep()));
            error_vec.push_back(std::to_string(env.get_L()));
            error_vec.push_back(std::to_string(my_experiment.get_int_param("run")));
            error_vec.push_back(std::to_string(running_error));
            error_logger.push_back(error_vec);
            current_seq_length = env.get_L();

        }
        if(counter % 50000 < 200)
        {
            std::vector<float> cur_state = env.get_state();
            cur_state.push_back(target);
            cur_state.push_back(my_network.read_output_values()[0]);
            cur_state.push_back(env.get_L());
            cur_state.push_back(env.get_data_timestep());
            print_vector(cur_state);
        }

//        if(counter%500000  == 499999)
//        {
//            my_network.add_memory(my_experiment.get_float_param("step_size"));
//            my_network.add_memory(my_experiment.get_float_param("step_size"));
//            my_network.add_memory(my_experiment.get_float_param("step_size"));
//            my_network.add_memory(my_experiment.get_float_param("step_size"));
//            my_network.add_memory(my_experiment.get_float_param("step_size"));
//
//
//        }

        if(counter % 100000 == 99998)
        {
//            print_vector(my_network.get_memory_weights());
            std::cout << "Pushing results" << std::endl;
            synapses_metric.add_values(error_logger);
            std::cout << "Results added " << std::endl;
            std::cout << "Len = " << error_logger.size() << std::endl;
            error_logger.clear();
        }
        if (counter % 10000 == 0 || counter % 10000 == 999 || counter % 10000 == 998) {
            std::cout << "### STEP = " << counter << std::endl;
            std::cout << "Running error = " << running_error << std::endl;
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
    graph_state.add_value(graph_data);

    return 0;
}

//
// Created by Khurram Javed on 2021-04-01.
//

