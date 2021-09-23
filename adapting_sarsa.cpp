//
// Created by taodav on 17/9/21.
//


#include "include/agents/sarsa.h"
#include "include/environments/mountain_car.h"
#include "include/nn/networks/adaptive_value_network.h"

int main(int argc, char *argv[]) {
  float gamma = 0.99;
  float lambda = 0.9;
  float step_size = powf(2, -5);
  int max_episode_steps = 1000;
  int total_steps = 1000000;
  int add_feature_every = 10000;
  int seed = 2020;
  int discretization = 30;
  float util_to_keep = 0.001;

  MountainCar env(seed + 2020, discretization);

  auto *my_network = new AdaptiveValueNetwork(env.observation_shape(), env.n_actions(), seed,
                                              step_size, 1e-3, true, util_to_keep);


  //  Initialize agent. Right now we only have our SarsaAgent available.
  SarsaAgent agent(my_network,
                   env.n_actions(),
                   0.1,
                   lambda);

  std::vector<float> episode_returns;
  std::vector<float> avg_eps_loss;

  int step = 0;
  int episode = 0;
  while (step < total_steps){
    float episode_rews = 0;
    float episode_loss = 0;

    Observation obs = env.reset();
    int t;
    for (t = 0; t < max_episode_steps; t++) {
      int action = agent.step(obs.observation);

      Observation new_obs = env.step(action);
      step++;

      bool is_terminal = new_obs.is_terminal;
      float reward = new_obs.reward;
      std::vector<float> next_state = new_obs.observation;

      episode_rews += reward;

      float loss;
      if (is_terminal) {
        loss = agent.post_step(action, next_state, reward, 0);
        episode_loss += loss;
        episode++;
        break;
      }

      loss = agent.post_step(action, next_state, reward, gamma);
      episode_loss += loss;
      obs = new_obs;

      if (step % add_feature_every == (add_feature_every - 1)) {
        my_network->collect_garbage();

        std::cout << "============== Adding feature ===============\n" << std::endl;
        my_network->add_feature(step_size, util_to_keep);
      }
    }
    episode_returns.push_back(episode_rews);
    float avg_loss = episode_loss / static_cast<float>(t);
    avg_eps_loss.push_back(avg_loss);
    std::cout << "Episode " << episode << " completed." << std::endl;
    std::cout << "Total steps: " << step << std::endl;
    std::cout << "Returns: " << episode_rews << std::endl;
    std::cout << "Avg episode loss: " << avg_loss << "\n" << std::endl;
  }
}