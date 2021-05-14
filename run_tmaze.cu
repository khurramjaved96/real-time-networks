#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <signal.h>
#include <random>

#include "include/utils.h"
#include "include/neural_networks/networks/test_network.h"
#include "include/neural_networks/neural_network.h"
#include "include/experiment/Experiment.h"
#include "include/neural_networks/utils.h"
#include "include/experiment/Metric.h"
#include "src/hybrid_code/queue.cu"
#include "include/environments/tmaze.h"
#include "include/animal_learning/tracecondioning.h"
#include "include/neural_networks/networks/test_network.h"

volatile sig_atomic_t someone_killed_me = 0;
void sigint(int sig){
    someone_killed_me = 1;
    std::cout << "\nSIGINT detected, saving results and killing..." << std::endl;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, sigint);

//    TMaze env = TMaze(0, 5);
//    Observation obs = env.reset();
//    std::vector<float> N = {1,0,0,0};
//    std::vector<float> E = {0,1,0,0};
//    std::vector<float> W = {0,0,1,0};
//    std::vector<float> S = {0,0,0,1};
//
//    std::vector<float> direction;
//    std::mt19937 mt(time(0));
//    auto dir_smp = std::uniform_int_distribution<int>(0,3);
//    std::vector<std::tuple<Observation, int, int>> allobs;
//    for (int i = 0; i < 100; i++) {
//        int dir = dir_smp(mt);
//        if (dir == 0)
//            direction = N;
//        else if (dir == 1)
//            direction = E;
//        else if (dir == 2)
//            direction = S;
//        else if (dir == 3)
//            direction = W;
//        allobs.push_back(std::tuple<Observation, int, int>{env.step(direction), dir, env.get_current_pos_in_corridor()});
//    }
//    __builtin_trap();
//
//   TODO it seems to work as intended but should check again probably
//   set print pretty on
//   set $i=0
//   p allobs[$i++]

    std::cout << "Program started \n";
    Experiment exp = Experiment(argc, argv);
    std::cout << "Experiment object created \n";
    int width = exp.get_int_param("width");

    Metric run_state_metric = Metric(exp.database_name, "run_states",
                                     std::vector<std::string>{"run", "state", "state_comments"},
                                     std::vector<std::string>{"int", "VARCHAR(10)", "VARCHAR(30)"},
                                     std::vector<std::string>{"run"});
    Metric observations_metric = Metric(exp.database_name, "run_metrics",
                                        std::vector<std::string>{"run", "step", "episode", "eps_step", "avg_reward",
                                                                 "corridor_len", "corridor_pos", "state", "qvalues",
                                                                 "reward", "new_features"},
                                        std::vector<std::string>{"int", "int", "int", "int", "real", "int", "int",
                                                                 "JSON", "JSON", "real", "int"},
                                        std::vector<std::string>{"run", "step"});
    Metric graph_state = Metric(exp.database_name, "network_graphs",
                                std::vector<std::string>{"run", "step", "graph_data"},
                                std::vector<std::string>{"int", "int", "MEDIUMTEXT"},
                                std::vector<std::string>{"run", "step"});

    //TODO add num inp and out as params here fixed for env
    CustomNetwork my_network = CustomNetwork(exp.get_float_param("step_size"),
                                             exp.get_int_param("width"),
                                             exp.get_int_param("num_layers"),
                                             exp.get_int_param("sparsity"),
                                             exp.get_int_param("seed"));

    TMaze env = TMaze(exp.get_int_param("seed"),
                      exp.get_int_param("tmaze_corridor_length"));

    //get a sequence of data for data-driven initialization
//    if (exp.get_bool_param("data_driven_initialization")){
//        std::vector<std::vector<float>> input_batch;
//        input_batch.reserve(500);
//        for(int temp=0; temp<500; temp++)
//            input_batch.push_back(env.step(1));
//        my_network.initialize_network(input_batch);
//        env.reset();
//    }

    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    auto start = std::chrono::steady_clock::now();

    float R = 0;
    float R_old = 0;
    float average_reward = 0;
    float accuracy = -1;
    int selected_action_idx_old = 0;
    float gamma = exp.get_float_param("gamma");
    bool prev_was_terminal = false;

    int timestep_since_feat_added = exp.get_int_param("features_min_timesteps");
    int total_new_features = 0;
    std::string state = "finished";
    std::string state_comments = "";
    std::vector<std::vector<std::string>> metric_logger;
    std::vector<std::vector<std::string>> graph_logger;
    std::mt19937 mt(exp.get_int_param("seed"));
    auto exploration_sampler = std::uniform_int_distribution<int>(0,100);
    auto rnd_action_sampler = std::uniform_int_distribution<int>(0,3);

    for (int counter = 0; counter < exp.get_int_param("steps"); counter++) {
        if(someone_killed_me){
            state = "killed";
            state_comments = "interrupt_sig";
            break;
        }

        //TODO consider whether we want to add feats from start
        timestep_since_feat_added -= 1;

        Observation current_obs = env.get_current_obs();
        R_old = R;
        R = current_obs.reward;

        my_network.set_input_values(current_obs.state);
        my_network.step();
        //TODO env step here
        std::vector<float> qvalues = my_network.read_output_values();
        std::vector<float> action(qvalues.size(), 0.0);
        std::vector<float> targets(qvalues.size(), 0.0);
        int selected_action_idx = 0;
        if (exploration_sampler(mt) < exp.get_float_param("epsilon")*100)
            selected_action_idx = rnd_action_sampler(mt);
        else
            selected_action_idx = std::distance(qvalues.begin(), std::max_element(qvalues.begin(), qvalues.end()));
        action[selected_action_idx] = 1;
        //update the gradient for only the old action since current one is for bootstrap
        if (prev_was_terminal){
            // if previous state was terminal state, we dont want next episode's values to propagate into it
            targets[selected_action_idx_old] = R_old;
            prev_was_terminal = false;
        }
        else
            targets[selected_action_idx_old] = R_old + gamma * qvalues[selected_action_idx];
        selected_action_idx_old = selected_action_idx;

        if (counter > 0){
            my_network.introduce_targets(targets);
            average_reward = 0.999 * average_reward + 0.001 * R;
            if (current_obs.is_terminal){
                prev_was_terminal = true;
                if (accuracy == -1)
                    accuracy = int(R==4);
                else
                    accuracy = 0.999 * accuracy + 0.001 * int(R==4);
            }
        }
        if(counter % 50000 < 50000)
        {
            std::vector<float> cur_state = current_obs.state;
            cur_state.push_back(current_obs.episode);
            cur_state.push_back(current_obs.timestep);
            cur_state.push_back(current_obs.reward);
            cur_state.push_back(average_reward);
            cur_state.push_back(accuracy);
            cur_state.push_back(env.get_current_pos_in_corridor());
            print_vector(cur_state);
            print_vector(qvalues);
        }

        //TODO the qvalues used to make this action belong to old state
        current_obs = env.step(action);

        if(isnan(qvalues[selected_action_idx])){
          state = "killed";
          state_comments = "nan_prediction";
          std::cout << "killing due to nans" << std::endl;
          break;
        }

        if(counter < 10){
            std::string g = my_network.get_viz_graph();
            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(counter));
            graph_data.push_back(std::to_string(exp.get_int_param("run")));
            graph_data.push_back(g);
            graph_logger.push_back(graph_data);
        }


        if(exp.get_bool_param("add_features") &&
           timestep_since_feat_added < 1)
        {
            total_new_features += exp.get_int_param("num_new_features");
            timestep_since_feat_added = exp.get_int_param("features_min_timesteps");
            for (int i = 0; i < exp.get_int_param("num_new_features"); i++)
                my_network.add_memory(exp.get_float_param("step_size"));

            std::cout << "\n Adding features..." << std::endl;

            std::string g = my_network.get_viz_graph();
            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(counter));
            graph_data.push_back(std::to_string(exp.get_int_param("run")));
            graph_data.push_back(g);
            graph_logger.push_back(graph_data);
        }

//        if(counter % 300000 == 299998)
//        {
//            if(exp.get_bool_param("add_features"))
//                print_vector(my_network.get_memory_weights());
//            std::cout << "Pushing results" << std::endl;
//            observations_metric.add_values(obs_logger);
//            graph_state.add_values(graph_logger);
//            std::cout << "Results added " << std::endl;
//            std::cout << "Len = " << error_logger.size() << std::endl;
//            error_logger.clear();
//            obs_logger.clear();
//            graph_logger.clear();
//        }
//        if (counter % 10000 == 0 || counter % 10000 == 999 || counter % 10000 == 998) {
//            std::cout << "### STEP = " << counter << std::endl;
//            std::cout << "Running error = " << running_error << std::endl;
//            std::cout << "Running accuracy = " << running_accuracy << std::endl;
//        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << 1000000 / (1+(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                            exp.get_int_param("steps")))
              << " fps" << std::endl;

    //observations_metric.add_values(obs_logger);
    std::string g = my_network.get_viz_graph();
    std::vector<std::string> graph_data;
    std::cout << g << std::endl;
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
