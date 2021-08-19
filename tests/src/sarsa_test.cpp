//
// Created by taodav on 9/8/21.
//
#include "../../include/agents/sarsa.h"
#include "../../include/environments/mountain_car.h"
#include "../../include/nn/networks/linear_function_approximator.h"
#include "../include/test_cases.h"

/**
 * Sarsa(lambda) with LFA test.
 * @return whether or not the average returns over 5 runs are greater than -200
 */

bool sarsa_lfa_mc_test() {
  std::cout << "RL experiment started." << std::endl;
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

