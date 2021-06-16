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
#include "include/neural_networks/networks/adaptive_network.h"
#include "include/neural_networks/neural_network.h"
#include "include/experiment/Experiment.h"
#include "include/neural_networks/utils.h"
#include "include/experiment/Metric.h"
#include "include/environments/tmaze.h"

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

    //TODO add num inp and out as params here fixed for env
    ContinuallyAdaptingNetwork my_network = ContinuallyAdaptingNetwork(exp.get_float_param("step_size"),
                                                                       exp.get_int_param("width"),
                                                                       exp.get_int_param("seed"));

    TMaze env = TMaze(exp.get_int_param("seed"),
                      exp.get_int_param("tmaze_corridor_length"),
                      exp.get_int_param("episode_length"),
                      exp.get_int_param("episode_gap"),
                      exp.get_bool_param("prediction_problem"));

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

    int max_episodes = exp.get_int_param("max_episodes");
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
        bool rndact = false;
        if(someone_killed_me){
            run_state = "killed";
            run_state_comments = "interrupt_sig";
            break;
        }
        no_op_step -= 1;
        no_grad_after_eps_gap -= 1;
        timestep_since_feat_added -= 1;

        Observation current_obs = env.get_current_obs();
        if(max_episodes != -1 && current_obs.episode > max_episodes)
            break;

        R_old = R;
        R = current_obs.reward;

        my_network.set_input_values(current_obs.state);
        my_network.step();
        float target = 0.0;
        int selected_action_idx = -1;
        std::vector<float> qvalues = my_network.read_output_values();
        std::vector<float> action(qvalues.size(), 0.0);
        std::vector<bool> no_grad(qvalues.size(), true);
        //update the gradient for only the old action since current one is for bootstrap
        no_grad[selected_action_idx_old] = false;
        if (prev_was_terminal){
            // if previous state was terminal state, we dont want next episode's values to propagate into it
            // this is the update for last state of the episode
            target = R_old;
            prev_was_terminal = false;
        }
        else if (state_old == state_cur && current_obs.is_terminal){
            // if we areinside the gap between the episodes)
            no_grad_after_eps_gap = 3;
        }
        else if (no_op_step != 0) {
            if (exploration_sampler(mt) < exp.get_float_param("epsilon")*100){
                rndact = true;
                int rnd_action = rnd_action_sampler(mt);
                //for prediction problem, allow choosing only between two actions at junction
                if (exp.get_bool_param("prediction_problem") && state_old == env.junction_state)
                    rnd_action <= 1 ? selected_action_idx = 0 : selected_action_idx = 3;
                //always go west in junction
                else if (exp.get_bool_param("prediction_problem"))
                    selected_action_idx = 2;
                else
                    selected_action_idx = rnd_action_sampler(mt);
            }
            else{
                if (exp.get_bool_param("prediction_problem") && state_old == env.junction_state)
                    qvalues[0] > qvalues[3] ? selected_action_idx = 0 : selected_action_idx = 3;
                else if (exp.get_bool_param("prediction_problem"))
                    selected_action_idx = 2;
                else
                    selected_action_idx = std::distance(qvalues.begin(), std::max_element(qvalues.begin(), qvalues.end()));
            }
            action[selected_action_idx] = 1;
            target = R_old + gamma * qvalues[selected_action_idx];
            //TODO change here
            //if (no_op_step != 0)

            if(std::isnan(qvalues[selected_action_idx])){
              run_state = "killed";
              run_state_comments = "nan_prediction";
              std::cout << "killing due to nans" << std::endl;
              break;
            }
        }

        if (no_grad_after_eps_gap == 2)
            my_network.reset_trace();

        if (counter > 0){
            if (no_op_step == 0 || no_grad_after_eps_gap > 0)
                no_grad = std::vector<bool>(qvalues.size(), true);
            if (current_obs.episode > 10000){
                std::cout << "{\tq:";
                print_vector(qvalues);
                std::cout << ",\to:";
                print_vector(current_obs.state);
                std::cout << ",\ta:"<< selected_action_idx;
                std::cout << ",\ta_old:"<< selected_action_idx_old;
                std::cout << ",\tt:"<< target;
                std::cout << ",\tr:"<< R_old;
                std::cout << ",\tg:";
                print_vector(no_grad);
                std::cout << ",\trnd:" << rndact;
                std::cout << ",\terr:" << ep_error;
                std::cout << ",\top:" << no_op_step;
                std::cout << "\t}\n";
            }
            //print_vector(std::vector<float>(qvalues.size(), target));
            ep_error += my_network.introduce_targets(std::vector<float>(qvalues.size(), target), gamma, lambda, no_grad);
            if (current_obs.is_terminal && state_old != state_cur){
                prev_was_terminal = true;
                if (accuracy == -1){
                    average_reward = current_obs.cmltv_reward;
                    accuracy = int(R==4);
                    error = ep_error;
                }
                else{
                    average_reward = 0.999 * average_reward + 0.001 * current_obs.cmltv_reward;
                    accuracy = 0.999 * accuracy + 0.001 * int(R==4);
                    error = 0.999 * error + 0.001 * ep_error;
                }
                ep_error = 0;
            }
        }
        if(current_obs.is_terminal && state_old != state_cur)
            std::cout<< "EP:" << current_obs.episode << "\t\tSteps:" << current_obs.timestep <<  "\t\tR:" << average_reward << "\t\tAcc:" << accuracy << "\t\tErr:" << error << "\t\tSyn:" << my_network.get_total_synapses() << std::endl;

        if (current_obs.is_terminal && state_old != state_cur && current_obs.episode % 10 == 9){
            std::vector<std::string> episode_data;
            episode_data.push_back(std::to_string(exp.get_int_param("run")));
            episode_data.push_back(std::to_string(counter));
            episode_data.push_back(std::to_string(current_obs.episode));
            episode_data.push_back(std::to_string(current_obs.timestep));
            episode_data.push_back(std::to_string(average_reward));
            episode_data.push_back(std::to_string(accuracy));
            episode_data.push_back(std::to_string(error));
            episode_logger.push_back(episode_data);

            std::vector<std::string> network_data;
            network_data.push_back(std::to_string(counter));
            network_data.push_back(std::to_string(current_obs.episode));
            network_data.push_back(std::to_string(exp.get_int_param("run")));
            network_data.push_back(std::to_string(my_network.get_total_synapses()));
            network_logger.push_back(network_data);
        }

        if(exp.get_bool_param("add_features") && timestep_since_feat_added < 1)// && current_obs.episode <= 35000)
        {
            timestep_since_feat_added = exp.get_int_param("features_min_timesteps");
            my_network.college_garbage();
            for (int i = 0; i < exp.get_int_param("num_new_features"); i++)
                my_network.add_feature(exp.get_float_param("step_size"));

            std::cout << "\n Adding features..." << std::endl;

            std::vector<std::string> graph_data;
            graph_data.push_back(std::to_string(exp.get_int_param("run")));
            graph_data.push_back(std::to_string(counter));
            graph_data.push_back(std::to_string(current_obs.episode));
            graph_data.push_back(my_network.get_viz_graph());
            graph_logger.push_back(graph_data);
        }

        if(counter % 100000 == 99998){
            episodic_metric.add_values(episode_logger);
            graph_state.add_values(graph_logger);
            network_size_metric.add_values(network_logger);

            episode_logger.clear();
            graph_logger.clear();
            network_logger.clear();
        }

        if (selected_action_idx != -1)// and current_obs.state != env.terminal_state)
            selected_action_idx_old = selected_action_idx;
        if (state_old != state_cur && !current_obs.is_terminal){
            //current_obs = env.step(env.get_no_op_action());
            //no_op_step = 2;
            current_obs = env.step(action);
        }
        else{
            current_obs = env.step(action);
        }
        state_old = state_cur;
        state_cur = current_obs.state;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in milliseconds for per steps: "
              << 1000000 / (1+(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                            exp.get_int_param("steps")))
              << " fps" << std::endl;

    graph_data.clear();
    graph_data.push_back(std::to_string(exp.get_int_param("run")));
    graph_data.push_back(std::to_string(exp.get_int_param("steps")));
    graph_data.push_back(std::to_string(env.get_current_obs().episode));
    graph_data.push_back(my_network.get_viz_graph());
    graph_logger.push_back(graph_data);

    episodic_metric.add_values(episode_logger);
    graph_state.add_values(graph_logger);
    network_size_metric.add_values(network_logger);

    std::vector<std::string> run_state_data;
    run_state_data.push_back(std::to_string(exp.get_int_param("run")));
    run_state_data.push_back(run_state);
    run_state_data.push_back(run_state_comments);
    run_state_metric.add_value(run_state_data);

    return 0;
}
