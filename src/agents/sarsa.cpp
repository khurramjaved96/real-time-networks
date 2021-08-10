//
// Created by taodav on 29/7/21.
//

#include <cmath>

#include "../../include/agents/sarsa.h"
#include "../../include/nn/networks/feedforward_state_value_network.h"


SarsaAgent::SarsaAgent(ContinuallyAdaptingNetwork *in_network, int n_actions, float epsilon, float lambda) {
  this->network = in_network;
  this->n_actions = n_actions;
  this->epsilon = epsilon;
  this->lambda = lambda;
  this->action_sampler = std::uniform_int_distribution<int>(0, this->n_actions);
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

  std::vector<float> q_values = this->network->read_output_values();
  std::vector<bool> no_grad_mask(q_values.size(), true);
  no_grad_mask[action] = false;

  //  Here we need to get q values of next_state
  std::vector<float> next_q_values = this->network->forward_pass_without_side_effects(next_state);

  //  Epsilon-greedy actions
  int next_action;
  if (this->exploration_sampler(mt) < this->epsilon){
    next_action = this->action_sampler(mt);
  } else {
    next_action = std::distance(next_q_values.begin(), std::max_element(next_q_values.begin(), next_q_values.end()));
  }

  float target = reward + gamma * next_q_values[next_action];

  this->network->introduce_targets(std::vector<float>(q_values.size(), target), gamma, this->lambda, no_grad_mask);

  return powf(target - q_values[action], 2);
}

/**
 * At the end of an episode, we need to propagate all the gradients
 * and assign all the credit. Finally we reset all
 */
void SarsaAgent::terminal() {
  bool credit_remaining = this->network->any_credit_remaining();
  std::vector<float> zeros;

  for (int j = 0; j < this->network->input_neurons.size(); j++) {
    zeros.push_back(0.0);
  }

  while (credit_remaining) {
    this->network->set_input_values(zeros);
    this->network->step();
    credit_remaining = this->network->any_credit_remaining();
  }
  this->network->reset_trace();
}