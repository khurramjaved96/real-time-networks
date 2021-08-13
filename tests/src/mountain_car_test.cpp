
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <random>

#include "../include/test_cases.h"
#include "../../include/utils.h"
#include "../../include/environments/mountain_car.h"
#include "../../include/nn/networks/linear_function_approximator.h"
#include "../../include/experiment/Experiment.h"
#include "../../include/nn/utils.h"


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


std::mt19937 mt(0);

int policy(std::vector<float> x) {
  if (x[1] < 0)
    return 0;
  else
    return 2;
}

std::vector<float> compute_monte_carlo_targets(std::vector<float> rewards, float gamma){
  std::vector<float> results;
  float running_sum = 0;
//  print_vector(rewards);
  for(int i = rewards.size()-1; i >= 0; i--){

    running_sum = running_sum*gamma + rewards[i];
    results.push_back(running_sum);
  }
  std::reverse(results.begin(),results.end());
  return results;
}

float compute_average_error(std::vector<float> x1, std::vector<float> x2){
  float sum_of_error = 0;
  if(x1.size() != x2.size()){
    std::cout << "Vectors not of the same shape\n";
    exit(1);
  }
  for(int i = 0; i<x1.size(); i++){
    sum_of_error += (x1[i] - x2[i])*(x1[i] - x2[i]);
  }
  sum_of_error /= x1.size();
  return sqrt(sum_of_error);
}


bool mountain_car_test() {

  ;

  float gamma = 1;
  float lambda = 0.9;

  // Initialize our dataset
  int input_feature_size = 30;
  MountainCar tc = MountainCar(0, input_feature_size);

  LinearFunctionApproximator my_network =
      LinearFunctionApproximator(input_feature_size * 2, tc.n_actions(), 3e-2,
                                 1e-3, true);

  float running_error;


  int episode = 0;
  int episode_return = 0;
  std::vector<float> episode_predictions;
  std::vector<float> rewards;
  while(episode < 80) {
    episode_return--;
    auto obs = tc.get_current_obs();

    my_network.set_input_values(obs.observation);
    my_network.step();
    auto targets = my_network.read_output_values();

    int action = policy(obs.state);
    tc.step(action);
    obs = tc.get_current_obs();
    rewards.push_back(obs.reward);
    float actual_prediction = targets[action];
    episode_predictions.push_back(actual_prediction);

    if (tc.at_goal()) {

      episode_return = 0;
      std::vector<float> monte_carlo_targets = compute_monte_carlo_targets(rewards, gamma);
//      print_vector(monte_carlo_targets);
//      print_vector(episode_predictions);
      float episode_error = compute_average_error(monte_carlo_targets, episode_predictions);
      if(episode_error < 15){
        return true;
      }
//      std::cout << "Episode no " << episode << std::endl;
//      std::cout << "Average msre error = " << episode_error << std::endl;
      if (episode % 100 == 99) {
//        std::cout << "Pred\tGT\n";
        for(int i = 0; i<monte_carlo_targets.size(); i++){
          std::cout << episode_predictions[i] << "\t" << monte_carlo_targets[i] <<"\n";
        }
      }
      episode_predictions.clear();
//      monte_carlo_targets.clear();
      rewards.clear();
//      exit(1);
      targets[action] = -1;

      float error = my_network.introduce_targets(targets, 0, lambda);
      tc.reset();
//      std::cout << "Epsilon = " << epsilon << std::endl;
      episode++;

    }
    else {
      auto next_predictions = my_network.forward_pass_without_side_effects(tc.get_current_obs().observation);
      int next_action = policy(tc.get_current_obs().state);
      float actual_prediction = targets[action];
      targets[action] = -1 + gamma * next_predictions[next_action];

      float error = my_network.introduce_targets(targets, gamma, lambda);
    }
  }
  return false;
}
