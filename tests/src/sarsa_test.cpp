//
// Created by taodav on 9/8/21.
//
#include "../../include/agents/sarsa.h"
#include "../../include/environments/mountain_car.h"
#include "../../include/nn/networks/linear_function_approximator.h"
#include "../include/test_cases.h"
#include "../../include/nn/networks/sarsa_lambda.h"
#include "../../include/utils.h"
#include <vector>

/**
 * Sarsa(lambda) with LFA test.
 * @return whether or not the average returns over 5 runs are greater than -200
 */


bool sarsa_lambda_test() {
  std::vector<std::vector<float>> states;
  auto network = SarsaLambda(4, 25, 1e-2, 1e-2);
  std::vector<int> indices{-1, 10, 11, -1, 7, -1, 13, 14};
  std::vector<int> actions{0, 1, 0, 1, 3, 1, 1, 0};
  std::vector<int> rewards{0, 0, 0, 0, 0, 0, 1, 0};
  std::vector<float> gammas{1, 1, 1, 1, 0.5, 0.5, 0, 1};
  std::vector<bool> terminal{false, false, false, false, false, false, true};

  for(int i = 0; i < 8; i++){
    std::vector<float> s(25, 0);
    if(indices[i] != -1)
      s[indices[i]] = 1;
    states.push_back(s);
  }
  for(int step = 0; step < 2000; step++){
    std::vector<float> s = states[step%8];
    int action = actions[step%8];
//    print_vector(network.forward_pass_without_side_effects(s));
    std::vector<float> v_s = network.forward(s);
    std::vector<float> s_prime = states[(step+1)%8];
    std::vector<float> v_s_prime = network.forward_pass_without_side_effects(s_prime);
    float reward = rewards[step%8];
    int a_prime = actions[(step + 1)%8];
    float gamma = gammas[step%8];
//    std::cout << " GAMMA VALUE HER EHERE HERE " << gamma << std::endl;
    float target_value = reward + gamma* v_s_prime[a_prime];
//    if(gamma == 0.0){
//      std::cout << "Target = " << target_value << std::endl;
//    }
    network.introduce_targets(target_value, gamma, 1, action);
    network.backward();

    network.update_utility();
    network.update_weights();
    }
  return true;
  }


bool sarsa_lfa_mc_test() {
//  std::cout << "RL experiment started." << std::endl;
  float gamma = 0.99;
  float lambda = 0.9;
  float step_size = powf(2, -5);
  int max_episode_steps = 1000;
  int episodes = 50;

  float avg_returns = 0;

  std::vector<int> seeds{0, 1, 2, 3, 4};


  for (int s = 0; s < seeds.size(); s++) {

    //  Initialize environment. Right now we only have Mountain Car available
    MountainCar env(seeds[s] + 2020, 30);

    auto *my_network = new LinearFunctionApproximator(env.observation_shape(), env.n_actions(),
                                                      step_size,
                                                      1e-3, false);

    //  Initialize agent. Right now we only have our SarsaAgent available.
    SarsaAgent agent(my_network,
                     env.n_actions(),
                     0.1,
                     lambda);
    float last_10 = 0;

    for (int ep = 0; ep < episodes; ep++){
      float episode_rews = 0;

      Observation obs = env.reset();

      for (int t = 0; t < max_episode_steps; t++) {
        int action = agent.step(obs.observation);

        Observation new_obs = env.step(action);

        bool is_terminal = new_obs.is_terminal;
        float reward = new_obs.reward;
        std::vector<float> next_state = new_obs.observation;

        episode_rews += reward;

        float loss;
        if (is_terminal) {
          loss = agent.post_step(action, next_state, reward, 0);
          break;
        }

        loss = agent.post_step(action, next_state, reward, gamma);
        obs = new_obs;
      }
      if ((ep + 1) > episodes - 10) {
        last_10 += episode_rews;
      }
    }
    avg_returns += (last_10 / 10);
  }
  return (avg_returns / seeds.size()) > -200;
}

