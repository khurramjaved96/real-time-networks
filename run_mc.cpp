//
// Created by taodav on 29/6/21.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <signal.h>
#include <random>

#include "include/environments/mountain_car.h"
#include "include/neural_networks/networks/adaptive_network.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"

int main(int argc, char *argv[]) {
    std::cout << "Starting Sarsa(lambda) on Mountain Car";

    Experiment exp = Experiment(argc, argv);

    Metric run_state_metric = Metric(exp.database_name, "run_states",
                                     std::vector<std::string>{"run", "run_state", "run_state_comments"},
                                     std::vector<std::string>{"int", "VARCHAR(10)", "VARCHAR(30)"},
                                     std::vector<std::string>{"run"});
    Metric episodic_metric = Metric(exp.database_name, "episodic_metrics",
                                    std::vector<std::string>{"run", "step", "episode", "timestep", "avg_reward", "accuracy", "error"},
                                    std::vector<std::string>{"int", "int", "int", "int", "real", "real", "real"},
                                    std::vector<std::string>{"run", "episode"});
    Metric graph_state = Metric(exp.database_name, "network_graphs",
                                std::vector<std::string>{"run", "step", "episode", "graph_data"},
                                std::vector<std::string>{"int", "int", "int", "MEDIUMTEXT"},
                                std::vector<std::string>{"run", "step", "episode"});

    Metric network_size_metric = Metric(exp.database_name, "network_metrics",
                                        std::vector<std::string>{"step", "episode", "run", "total_synapses"},
                                        std::vector<std::string>{"int", "int", "int", "int"},
                                        std::vector<std::string>{"step", "episode", "run"});

    MountainCar env = MountainCar(exp.get_int_param("seed"), false);

    ContinuallyAdaptingNetwork my_network = ContinuallyAdaptingNetwork(exp.get_float_param("step_size"),
                                                                       env.observation_shape(),
                                                                       env.n_actions(),
                                                                       exp.get_int_param("width"),
                                                                       exp.get_int_param("seed"));

    std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
    my_network.viz_graph();
    std::vector<std::string> graph_data;
    graph_data.push_back(std::to_string(exp.get_int_param("run")));
    graph_data.push_back(std::to_string(0));
    graph_data.push_back(std::to_string(0));
    graph_data.push_back(my_network.get_viz_graph());
    graph_state.add_value(graph_data);

    auto start = std::chrono::steady_clock::now();

    float R = 0;
    float R_old = 0;
    float average_reward = -1;
    float accuracy = -1;
    float error = -1;
    float ep_error = 0;
    int selected_action_idx_old = 0;
    float gamma = exp.get_float_param("gamma");
// TODO new param
    float lambda = exp.get_float_param("lambda");
    bool prev_was_terminal = false;
    int no_op_step = 0;
    int no_grad_after_eps_gap = 0;
    std::vector<float> state_cur = env.get_current_obs().state;
    std::vector<float> state_old = env.get_current_obs().state;

    int timestep_since_feat_added = exp.get_int_param("features_min_timesteps");

    std::string run_state = "finished";
    std::string run_state_comments = "";
    std::vector<std::vector<std::string>> network_logger;
    std::vector<std::vector<std::string>> episode_logger;
    std::vector<std::vector<std::string>> graph_logger;
    std::mt19937 mt(exp.get_int_param("seed"));
    auto exploration_sampler = std::uniform_int_distribution<int>(0,100);
    auto rnd_action_sampler = std::uniform_int_distribution<int>(0,3);

    for (int counter = 0; counter < exp.get_int_param("steps"); counter++) {
        Observation current_obs = env.get_current_obs();
        R_old = R;
        R = current_obs.reward;
        my_network.set_input_values(current_obs.state);
        my_network.step();
        std::vector<float> qvalues = my_network.read_output_values();
        std::vector<bool> no_grad(qvalues.size(), true);

//      Only update the head of the last action
        no_grad[selected_action_idx_old] = false;
        int action = 0;
        if (exploration_sampler(mt) < exp.get_float_param("epsilon")*100){
            action = env.get_random_action();
        } else {
            action = std::distance(qvalues.begin(), std::max_element(qvalues.begin(), qvalues.end()));
        }
        float target;
        if (prev_was_terminal) {
            target = R_old;
        } else {
            target = R_old + gamma * qvalues[action];
        }
        if(std::isnan(qvalues[action])){
            run_state = "killed";
            run_state_comments = "nan_prediction";
            std::cout << "killing due to nans" << std::endl;
            break;
        }
        if (counter > 0) {
//          TODO: maybe some logging?
            ep_error += my_network.introduce_targets(std::vector<float>(qvalues.size(), target), gamma, lambda, no_grad);

            if (prev_was_terminal) {
//              If our previous action resulted in termination
//              then we need to reset everything and select a new action
                env.reset();

                my_network.reset_all_less_weights();
            }
            if (current_obs.is_terminal) {
                prev_was_terminal = true;

            }
        }

        if(counter % 100000 == 99998){
            episodic_metric.add_values(episode_logger);
            graph_state.add_values(graph_logger);
            network_size_metric.add_values(network_logger);

            episode_logger.clear();
            graph_logger.clear();
            network_logger.clear();
        }

        env.step(action);
        selected_action_idx_old  = action;
    }
}
