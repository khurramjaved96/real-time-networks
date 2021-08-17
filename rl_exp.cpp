//
// Created by taodav on 9/8/21.
//
#include "include/experiment/Experiment.h"
#include "include/agents/sarsa.h"
#include "include/environments/mountain_car.h"
#include "include/nn/networks/linear_function_approximator.h"

/**
 * Main entry point for running Mountain Car Experiments.
 * @param argc
 * @param argv
 * --run (int, 0), the run number
 * --gamma (float, 0.99), discount factor.
 * --epsilon (float, 0.1), epsilon-greedy exploration.
 * --lambda (float, 0.00), parameter for eligibility trace
 * --seed (int, 2021), what seed do we use?
 * --max_episodic_steps (int, 200), what is the max number of steps per episode?
 * --step_size (float, 0.0001), step size parameter.
 * --discretization (int, 10), how many bins for each dimension?
 * --steps (int, 100000), total number of steps we take in the environment
 * For now we only support MountainCar with Sarsa(lambda).
 * @return
 */
int main(int argc, char *argv[]) {
  Experiment exp(argc, argv);
  std::cout << "RL experiment started." << std::endl;
  float gamma = exp.get_float_param("gamma");
  float lambda = exp.get_float_param("lambda");
  int seed = exp.get_int_param("seed");

//  Initialize environment. Right now we only have Mountain Car available
  MountainCar env(seed, exp.get_int_param("discretization"));

//  Initialize network
//  auto *my_network = new ContinuallyAdaptingNetwork(exp.get_float_param("step_size"),
//                                        seed, env.observation_shape(), env.n_actions());

  auto *my_network = new LinearFunctionApproximator(env.observation_shape(), env.n_actions(),
                                                    exp.get_float_param("step_size"),
                                                    1e-3, false);

//  Initialize agent. Right now we only have our SarsaAgent available.
  SarsaAgent agent(my_network,
                   env.n_actions(),
                   exp.get_float_param("epsilon"),
                   lambda);
  int tstep = 0;
  int episode = 0;
  while (tstep < exp.get_int_param("steps")){
    float episode_rews = 0;
    float episode_loss = 0;
    int ep_timesteps;

    Observation obs = env.reset();

    for (int t = 0; t < exp.get_int_param("max_episodic_steps"); t++) {
      int action = agent.step(obs.observation);

//      if (obs.state[1] < 0) {
//          action = 0;
//      } else {
//          action = 2;
//      }

      Observation new_obs = env.step(action);
      tstep++;

      bool is_terminal = new_obs.is_terminal;
      float reward = new_obs.reward;
      std::vector<float> next_state = new_obs.observation;

      episode_rews += reward;
      ep_timesteps = t;

      if (is_terminal) {
        episode_loss += agent.post_step(action, next_state, reward, 0);
        break;
      }

      episode_loss += agent.post_step(action, next_state, reward, gamma);
      obs = new_obs;
    }
    episode++;


//   Logging after an episode
    std::cout << "### STEP = " << tstep << std::endl;
    std::cout << "Episode num = " << episode << std::endl;
    std::cout << "Episode return = " << episode_rews << std::endl;
    std::cout << "Episode timesteps = " << ep_timesteps << std::endl;
    std::cout << "Avg loss = " << episode_loss / (ep_timesteps + 1) << std::endl;
    std::cout << "Total elements = " << my_network->all_heap_elements.size() << std::endl;
    std::cout << "Output synapses = " << my_network->output_synapses.size() << std::endl;
    std::cout << "Total synapses = " << my_network->all_synapses.size() << std::endl;
    std::cout << "Output Neurons = " << my_network->output_neurons.size() << "\t"
              << my_network->input_neurons.size() << std::endl;
    std::cout << "Total Neurons = " << my_network->all_neurons.size() << std::endl;

  }

}

