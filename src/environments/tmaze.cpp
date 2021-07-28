#include "../../include/environments/tmaze.h"
#include <math.h>
#include <random>
#include <vector>
#include "../../include/utils.h"

TMaze::TMaze(int seed, int length_of_corridor, int episode_length, int episode_gap, bool prediction_problem)
    : mt(seed) {
  // detailed desc in header
  this->current_episode = 1;
  this->length_of_corridor = length_of_corridor;
  this->direction_sampler = std::uniform_int_distribution<int>(0, 1);
  this->action_sampler = std::uniform_int_distribution<int>(0, 3);
  this->current_obs = this->reset();
  this->prediction_problem = prediction_problem;
  this->episode_length = episode_length;
  this->episode_gap = episode_gap;
  this->current_episode_pos_in_gap = 0;
}

int TMaze::get_current_pos_in_corridor() {
  return this->current_pos_in_corridor;
}

int TMaze::get_length_of_corridor() {
  return this->length_of_corridor;
}

void TMaze::set_length_of_corrider(int value) {
  if (this->current_obs.state != this->direction_state) {
    std::cout << "Error: Attempted to change the corridor length in middle of episode" << std::endl;
    exit(1);
  }
  this->length_of_corridor = value;
}

Observation TMaze::get_current_obs() {
  return this->current_obs;
}

std::vector<float> TMaze::generate_direction_state() {
  float dir_bit = static_cast<float>((direction_sampler(mt)));
  return std::vector < float > {dir_bit, 1, 1 - dir_bit};
}

std::vector<float> TMaze::get_no_op_action() {
  return this->no_op;
}

std::vector<float> TMaze::get_random_action() {
  // TODO exclude the greedy action mby? Should I?
  std::vector<float> random_action(4, 0.0);
  int action_idx = action_sampler(mt);
  random_action[action_idx] = 1;
  return random_action;
}

std::vector<float> TMaze::get_optimal_action(std::vector<float> action) {
  // Turns it into a prediction problem of delay length_of_corridor
  // the agent always passes through the corridor smoothly
  if (this->current_obs.state == this->junction_state)
    return action;
  else
    return this->W;
}

Observation TMaze::reset() {
  Observation obs;
  obs.timestep = 0;
  obs.episode = this->current_episode;
  obs.reward = 0;
  obs.cmltv_reward = 0;
  obs.is_terminal = false;
  obs.state = this->generate_direction_state();
  // print_vector(obs.state);
  this->current_obs = obs;

  this->direction_state = obs.state;
  this->current_pos_in_corridor = 0;

  if (this->direction_state[0] == 1)
    this->correct_direction = this->N;
  else
    this->correct_direction = this->S;

  this->current_episode_pos_in_gap = 0;
  return obs;
}

Observation TMaze::step(std::vector<float> action) {
  // actions: N[1000], S[0001], E[0100], W[0010]
  // if (this->prediction_problem)
  //    action = this->get_optimal_action(action);

  if (this->current_obs.timestep > this->episode_length)
    this->current_obs.is_terminal = true;

  if (this->current_episode_pos_in_gap < this->episode_gap && this->current_obs.is_terminal) {
    this->current_episode_pos_in_gap += 1;
    this->current_obs.reward = 0;
    return this->current_obs;
  }

  if (this->current_obs.is_terminal) {
    this->current_episode += 1;
    return this->reset();
  }

  if (action == this->no_op)
    return this->current_obs;

  this->current_obs.timestep += 1;
  this->current_obs.is_terminal = false;

  if ((this->current_pos_in_corridor == 0 && action == this->E) ||
      (this->current_pos_in_corridor < this->length_of_corridor && (action == this->N || action == this->S)) ||
      (this->current_pos_in_corridor == this->length_of_corridor && action == this->W)) {
    // if the agent stands still, -ve reward and no state change
    this->current_obs.reward = -0.1;
  } else if (this->current_pos_in_corridor == 1 && action == this->E) {
    // if the agent goes back to the start state from corridor
    this->current_pos_in_corridor -= 1;
    this->current_obs.reward = 0;
    this->current_obs.state = this->direction_state;
  } else if (
      (this->current_pos_in_corridor < (this->length_of_corridor - 1) && (action == this->W || action == this->E)) ||
          (this->current_pos_in_corridor == (this->length_of_corridor - 1) && action == this->E)) {
    // if the agent is moving across the corridor
    if (action == this->E)
      this->current_pos_in_corridor -= 1;
    else
      this->current_pos_in_corridor += 1;
    this->current_obs.reward = 0;
    this->current_obs.state = this->corridor_state;
  } else if (this->current_pos_in_corridor == (this->length_of_corridor - 1) && action == this->W) {
    // if the agent moves from corridor state -> junction state
    this->current_pos_in_corridor += 1;
    this->current_obs.reward = 0;
    this->current_obs.state = this->junction_state;
  } else if (this->current_pos_in_corridor == this->length_of_corridor && action == this->E) {
    // if the agent moves from junction state -> corridor state
    this->current_pos_in_corridor -= 1;
    this->current_obs.reward = 0;
    this->current_obs.state = this->corridor_state;
  } else if (this->current_pos_in_corridor == this->length_of_corridor && action == this->correct_direction) {
    // if the agent is at junction and picks the correct action
    this->current_obs.reward = 4;
    this->current_obs.state = this->terminal_state;
    this->current_obs.is_terminal = true;
  } else if (this->current_pos_in_corridor == this->length_of_corridor && action != this->correct_direction) {
    // if the agent is at junction and picks the wrong action
    // NOTE: this is not the same as paper. Paper uses -0.1 but I use -4 here since we dont
    // have directed exploration. It takes really long to learn without it.
    this->current_obs.reward = -0.1;
    this->current_obs.state = this->terminal_state;
    this->current_obs.is_terminal = true;
  } else {
    std::cout << "Error: Unhandled state/action pair" << std::endl;
    exit(1);
  }
  this->current_obs.cmltv_reward += this->current_obs.reward;
  return this->current_obs;
}
