//
// Created by Khurram Javed on 2021-09-17.
//

#ifndef INCLUDE_ENVIRONMENTS_SUPERVISED_IMPRINTING_H_
#define INCLUDE_ENVIRONMENTS_SUPERVISED_IMPRINTING_H_

#include <vector>
#include <random>

class SupervisedImprintingEnv{
  int input_features;
  int distribution_size;
  int current_index;
  std::uniform_int_distribution<int> index_sampler;
  std::mt19937 mt;
  std::vector<float> targets;
  std::vector<std::vector<float>> features;

 public:
  SupervisedImprintingEnv(int seed);
  std::vector<float> create_pattern();
  float create_target();
  void replace_random_pattern();
  std::vector<float> get_x();
  std::vector<float> get_y();
  void step();
  int get_index();

};

#endif  // INCLUDE_ENVIRONMENTS_SUPERVISED_IMPRINTING_H_
