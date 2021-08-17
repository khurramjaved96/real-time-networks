//
// Created by taodav on 29/7/21.
//

#include <cmath>

#include "../../include/utils.h"
#include "../../include/agents/sarsa.h"
#include "../../include/nn/networks/feedforward_state_value_network.h"


SarsaAgent::SarsaAgent(Network *in_network, int n_actions, float epsilon, float lambda) {
  this->network = in_network;
  this->n_actions = n_actions;
  this->epsilon = epsilon;
  this->lambda = lambda;
  this->action_sampler = std::uniform_int_distribution<int>(0, this->n_actions - 1);
  this->steps = 0;

  this->exploration_sampler = std::uniform_real_distribution<float>(0,1);
}

void SarsaAgent::set_eps(float eps) {
  this->epsilon = eps;
}

/**
 * Given a state, propagates the input forward a step and updates our network.
 * Returns the action.
 * @return action for the current time step.
 */
int SarsaAgent::step(std::vector<float> state) {

  //  This is assuming state is propagated right away.
  this->network->set_input_values(state);
  this->network->step();

  std::vector<float> q_values = this->network->read_output_values();
  int action;
  if (this->exploration_sampler(mt) < this->epsilon){
    action = this->action_sampler(mt);
  } else {
    action = std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
  }

  this->steps++;

  return action;
}

/**
 * Post step processing.
 * Introduces targets for our network.
 * @param action
 * @param next_state
 * @param reward
 * @return
 */
float SarsaAgent::post_step(int action, std::vector<float> next_state, float reward, float gamma) {

  std::vector<float> targets = this->network->read_output_values();

  float q_value = targets[action];
  std::vector<float> next_q_values(targets.size(), 0);

  //  Here we need to get q values of next_state
  if (gamma > 0) {
    next_q_values = this->network->forward_pass_without_side_effects(next_state);
  }

  //  Epsilon-greedy actions
  int next_action;
  if (this->exploration_sampler(mt) < this->epsilon){
    next_action = this->action_sampler(mt);
  } else {
    next_action = std::distance(next_q_values.begin(), std::max_element(next_q_values.begin(), next_q_values.end()));
  }

  float target = reward + gamma * next_q_values[next_action];
  targets[action] = target;

  this->network->introduce_targets(targets, gamma, this->lambda);

  float loss = powf(target - q_value, 2);

  return loss;
}
