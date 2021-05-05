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
//    std::string default_config = "--name test --width 10 --seed 0 --steps 100 --run 0 --step_size 0.0001 --num_layers 5 --data_driven_initialization false --randomize_sequence_length true --add_features false --sparsity 98 --sequence_gap 20"
//    __builtin_trap();
    std::cout << "Program started \n";
    Experiment exp = Experiment(argc, argv);
    std::cout << "Experiment object created \n";
    int width = exp.get_int_param("width");

    Metric synapses_metric = Metric(exp.database_name, "error_table",
                                    std::vector<std::string>{"step", "datatime", "seq_length", "run", "error", "accuracy"},
                                    std::vector<std::string>{"int", "int", "int", "int", "real", "real"},
                                    std::vector<std::string>{"step",  "run" });
    Metric run_state_metric = Metric(exp.database_name, "state_table",
                                     std::vector<std::string>{"run", "state", "state_comments"},
                                     std::vector<std::string>{"int", "MEDIUMTEXT", "MEDIUMTEXT"},
                                     std::vector<std::string>{"run"});
    Metric observations_metric = Metric(exp.database_name, "obs_table",
                                        std::vector<std::string>{"run", "step", "inp_start_flag", "inp_end_flag", "stml_seq", "target", "pred", "L", "seq_len", "data_timestep"},
                                        std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "int", "int", "int"},
                                        std::vector<std::string>{"run", "step"});
    Metric graph_state = Metric(exp.database_name, "graph",
                                std::vector<std::string>{"step", "run", "graph_data"},
                                std::vector<std::string>{"int", "int", "MEDIUMTEXT"},
                                std::vector<std::string>{"step", "run"});

    CustomNetwork my_network = CustomNetwork(exp.get_float_param("step_size"),
                                             exp.get_int_param("width"),
                                             exp.get_int_param("num_layers"),
                                             exp.get_int_param("sparsity"),
                                             exp.get_int_param("seed"));

    CopyTask env = CopyTask(exp.get_int_param("seed"),
                            exp.get_bool_param("randomize_sequence_length"),
                            exp.get_int_param("sequence_gap"));
    //get a sequence of data for data-driven initialization
    if (exp.get_bool_param("data_driven_initialization")){
        std::vector<std::vector<float>> input_batch;
        input_batch.reserve(500);
        for(int temp=0; temp<500; temp++)
            input_batch.push_back(env.step(1));
        my_network.initialize_network(input_batch);
        env.reset();
    }

    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    auto start = std::chrono::steady_clock::now();

    float running_error = -1;
    float running_accuracy = -1;
    float target = 0;
    float prediction = 0;
    float last_err = 1;
    float target_old = 0;
    int current_seq_length = 1;
    std::string state = "finished";
    std::string state_comments = "";
    std::vector<std::vector<std::string>> error_logger;
    std::vector<std::vector<std::string>> state_logger;
    std::vector<std::vector<std::string>> graph_logger;

    std::cout << "Flag Bit \t Pred Bit \t Target \t Pred \t Seq_len \t Datatime" << std::endl;
    for (int counter = 0; counter < exp.get_int_param("steps"); counter++) {

        auto state_current = env.step(last_err);
//        print_vector(state_current);
        my_network.set_input_values(state_current);
        my_network.step();
        prediction = my_network.read_output_values()[0];
        if(isnan(prediction)){
          state = "killed";
          state_comments = "nan_prediction";
          break;
        }

        target_old = target;
        target = env.get_target();
        if(counter > 0)
            my_network.introduce_targets(std::vector<float>{target_old});

        float error = (prediction - target)*(prediction - target);
        last_err = error;
        float accuracy;
        (prediction > 0.5 && target == 1) || (prediction < 0.5 && target == 0) ? accuracy=1 : accuracy=0;
        if (running_error == -1){
            running_error = error;
            running_accuracy = accuracy;
        }
        else{
            running_error = running_error * 0.999 + 0.001 *error;
            running_accuracy = running_accuracy * 0.999 + 0.001 *accuracy;
        }



        if(env.get_L() > current_seq_length || counter % 300 == 0)
        {
            std::vector<std::string> error_vec;
            error_vec.push_back(std::to_string(counter));
            error_vec.push_back(std::to_string(env.get_data_timestep()));
            error_vec.push_back(std::to_string(env.get_L()));
            error_vec.push_back(std::to_string(exp.get_int_param("run")));
            error_vec.push_back(std::to_string(running_error));
            error_vec.push_back(std::to_string(running_accuracy));
            error_logger.push_back(error_vec);
            current_seq_length = env.get_L();
        }


        std::vector<float> cur_state = env.get_state();

        if(counter < 1000 || counter > exp.get_int_param("steps") - 1000)
        {
            std::vector<std::string> state_vec;
            state_vec.push_back(std::to_string(exp.get_int_param("run")));
            state_vec.push_back(std::to_string(counter));
            state_vec.push_back(std::to_string(cur_state[0]));
            state_vec.push_back(std::to_string(cur_state[1]));
            state_vec.push_back(std::to_string(cur_state[2]));
            state_vec.push_back(std::to_string(target));
            state_vec.push_back(std::to_string(prediction));
            state_vec.push_back(std::to_string(env.get_L()));
            state_vec.push_back(std::to_string(env.get_seq_length()));
            state_vec.push_back(std::to_string(env.get_data_timestep()));
            state_logger.push_back(state_vec);
        }

        if(counter < 10){
            std::string g = my_network.get_viz_graph();
            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(exp.get_int_param("steps")));
            graph_data.push_back(std::to_string(exp.get_int_param("run")));
            graph_data.push_back(g);
            graph_logger.push_back(graph_data);
        }

        if(counter % 50000 < 200)
        {
            std::vector<float> cur_state = env.get_state();
            cur_state.push_back(target);
            cur_state.push_back(prediction);
            cur_state.push_back(env.get_L());
            cur_state.push_back(env.get_data_timestep());
            print_vector(cur_state);
        }

        if(counter%500000  == 499999 && exp.get_bool_param("add_features"))
        {
            my_network.add_memory(exp.get_float_param("step_size"));
            my_network.add_memory(exp.get_float_param("step_size"));
            my_network.add_memory(exp.get_float_param("step_size"));
            my_network.add_memory(exp.get_float_param("step_size"));
            my_network.add_memory(exp.get_float_param("step_size"));

            std::string g = my_network.get_viz_graph();
            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(exp.get_int_param("steps")));
            graph_data.push_back(std::to_string(exp.get_int_param("run")));
            graph_data.push_back(g);
            graph_logger.push_back(graph_data);
        }

        if(counter % 100000 == 99998)
        {
//            print_vector(my_network.get_memory_weights());
            std::cout << "Pushing results" << std::endl;
            synapses_metric.add_values(error_logger);
            observations_metric.add_values(state_logger);
            std::cout << "Results added " << std::endl;
            std::cout << "Len = " << error_logger.size() << std::endl;
            error_logger.clear();
            state_logger.clear();
        }
        if (counter % 10000 == 0 || counter % 10000 == 999 || counter % 10000 == 998) {
            std::cout << "### STEP = " << counter << std::endl;
            std::cout << "Running error = " << running_error << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << 1000000 / (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                            exp.get_int_param("steps"))
              << " fps" << std::endl;


    std::string g = my_network.get_viz_graph();
    std::vector<std::string> graph_data;
    graph_data.push_back(std::to_string(exp.get_int_param("steps")));
    graph_data.push_back(std::to_string(exp.get_int_param("run")));
    graph_data.push_back(g);
    graph_logger.push_back(graph_data);
    graph_state.add_values(graph_logger);

    std::vector<std::string> state_data;
    state_data.push_back(std::to_string(exp.get_int_param("run")));
    state_data.push_back(state);
    state_data.push_back(state_comments);
    run_state_metric.add_value(state_data);

    return 0;
}

//
// Created by Khurram Javed on 2021-04-01.
//

