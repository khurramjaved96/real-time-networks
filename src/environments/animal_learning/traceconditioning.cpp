//
// Created by Khurram Javed on 2021-04-22.
//


#include <math.h>
#include <random>
#include <vector>
#include "../../../include/environments/animal_learning/tracecondioning.h"

TraceConditioning::TraceConditioning(std::pair<int, int> ISI, std::pair<int, int> ISI_long, std::pair<int, int> ITI,
                                     int num_distractors, int seed) : ISI_sampler(ISI.first, ISI.second),
                                                                      ISI_long_sampler(ISI_long.first, ISI_long.second),
                                                                      ITI_sampler(ITI.first, ITI.second), mt(seed),
                                                                      NoiseSampler(0, 1) {
  this->num_distractors = num_distractors;
  for (int temp = 0; temp < 2; temp++) {
    current_state.push_back(0);
  }
  requires_reset = true;
  remaining_steps = 0;
  remaining_until_US = 0;
  ISI_length = ISI.first;
}

//
void TraceConditioning::increase_ISI(int t) {
  ISI_length += t;
  this->ISI_sampler = std::uniform_int_distribution<int>(ISI_length, ISI_length);
}

std::vector<float> TraceConditioning::get_state() {
  return this->current_state;
}

std::vector<float> TraceConditioning::step() {
  if (remaining_steps == 0) {
//        std::cout << "Gets here 6\n";
    return this->reset();
  }
  set_noise_bits();
  if (this->remaining_until_US == 1) {
    this->current_state[1] = 1;
  } else {
    this->current_state[1] = 0;
  }

  this->current_state[0] = 0;
  this->remaining_until_US--;
  this->remaining_steps--;
  return current_state;
}

std::vector<float> TraceConditioning::reset() {
  this->remaining_until_US = ISI_sampler(mt);
  this->remaining_steps = this->remaining_until_US + ITI_sampler(mt);

  set_noise_bits();
  current_state[0] = 1;  // Setting the CS
  current_state[1] = 0;  // setting the US
  return current_state;
}

void TraceConditioning::set_noise_bits() {
//    for (int temp = 3; temp < this->current_state.size(); temp++) {
//        if (NoiseSampler(mt) > 0.90 and false) {
//            this->current_state[temp] = 1;
//        } else {
//            this->current_state[temp] = 0;
//        }
//    }
}

float TraceConditioning::get_US() {
  return this->current_state[1];
}

float TraceConditioning::get_target(float gamma) {
  if (this->remaining_until_US > 0) {
    return pow(gamma, this->remaining_until_US-1);
  }
  return 0;
}
