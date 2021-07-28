//
// Created by Khurram Javed on 2021-04-22.
//

#include <iostream>
#include <vector>
#include <random>
#include <utility>

#ifndef INCLUDE_ANIMAL_LEARNING_TRACECONDIONING_H_
#define INCLUDE_ANIMAL_LEARNING_TRACECONDIONING_H_

class TraceConditioning {
//    std::pair<int, int> ISI;
//    std::pair<int, int> ITI;
  int num_distractors;
  std::mt19937 mt;
  std::uniform_int_distribution<int> ISI_sampler;
  std::uniform_int_distribution<int> ISI_long_sampler;
  std::uniform_int_distribution<int> ITI_sampler;
  std::uniform_real_distribution<float> NoiseSampler;
  std::vector<float> current_state;
  bool requires_reset;
  int remaining_steps;
  int ISI_length;
  int remaining_until_US;
  int remaining_until_US_long;

  void set_noise_bits();

 public:
  TraceConditioning(std::pair<int, int> ISI, std::pair<int, int> ISI_long, std::pair<int, int> ITI,
                    int num_distractors, int seed);

  std::vector<float> get_state();

  std::vector<float> step();

  std::vector<float> reset();

  float get_US();

  float get_target(float gamma);

  float cumulant();

  void increase_ISI(int t);
};

class TracePatterning {
  int num_distractors;
  std::mt19937 mt;
  std::uniform_int_distribution<int> ISI_sampler;
  std::uniform_int_distribution<int> ISI_long_sampler;
  std::uniform_int_distribution<int> ITI_sampler;
  std::uniform_real_distribution<float> NoiseSampler;
  std::vector<float> current_state;
  bool requires_reset;
  bool turn_off_first;
  int remaining_steps;
  int pattern_len;
  int ISI_length;
  int remaining_until_US;
  int remaining_until_US_long;
  bool valid;
  std::vector<float> distribution;
  std::vector<std::vector<float>> valid_patterns;

  void set_noise_bits();

 public:
  TracePatterning(std::pair<int, int> ISI, std::pair<int, int> ISI_long, std::pair<int, int> ITI, int num_distractors,
                  int seed);

  std::vector<float> get_state();

  std::vector<float> step();

  std::vector<float> reset();

  void remove_first_US();

  float get_US();

  float get_long_US();

  float get_target(float gamma);

  float get_target_long(float gamma);

  std::vector<float> create_pattern();

  float cumulant();

  void increase_ISI(int t);
};

#endif  // INCLUDE_ANIMAL_LEARNING_TRACECONDIONING_H_
