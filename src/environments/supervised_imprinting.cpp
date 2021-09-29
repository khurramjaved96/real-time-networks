//
// Created by Khurram Javed on 2021-04-22.
//

#include <math.h>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <execution>
#include "../../include/utils.h"
#include "../../include/environments/supervised_imprinting.h"


int DISTRIBUTION_SIZE = 30;
int INPUT_FEATURES = 10;
float PROB_FLIP = 0.001;

SupervisedImprintingEnv::SupervisedImprintingEnv(int seed) : mt(seed), index_sampler(0, DISTRIBUTION_SIZE - 1) {
  this->input_features = INPUT_FEATURES;
  this->distribution_size = DISTRIBUTION_SIZE;
  this->current_index = 0;
  while (this->features.size() < this->distribution_size) {
    std::vector<float> p = create_pattern();
    bool flag = true;
    for (auto it : this->features) {
      if (p == it) {
        flag = false;
      }
    }
    if (flag) {
      this->features.push_back(p);
      this->targets.push_back(this->create_target());
    }
  }
}

float SupervisedImprintingEnv::create_target() {
  std::uniform_real_distribution<float> temp_sampler(-10, 10);
  return temp_sampler(this->mt);
}

void SupervisedImprintingEnv::replace_random_pattern() {
  int index_to_replace = this->index_sampler(this->mt);
//  std::cout << "Index to replace\t" << index_to_replace << std::endl;
//  print_vector(this->features[index_to_replace]);
//  std::cout << "Target = " << this->targets[index_to_replace];
  bool not_replaced = true;
  while (not_replaced) {
    auto p = create_pattern();
    if (p != this->features[index_to_replace]) {
      this->features[index_to_replace] = p;
      this->targets[index_to_replace] = this->create_target();
      not_replaced = false;
    }
  }
//  std::cout << "AFTER\n";
//  std::cout << "Index to replace\t" << index_to_replace << std::endl;
//  print_vector(this->features[index_to_replace]);
//  std::cout << "Target = " << this->targets[index_to_replace];
}

int SupervisedImprintingEnv::get_index() {
  return this->current_index;
}

void SupervisedImprintingEnv::step() {
  this->current_index = this->index_sampler(this->mt);
  std::uniform_real_distribution<float> prob(0, 1);
  if(prob(this->mt) < PROB_FLIP)
  {
    this->replace_random_pattern();
//    std::cout << "Pattern replaced\n";
//    exit(1);
//
  }
}

std::vector<float> SupervisedImprintingEnv::get_x() {
  return this->features[this->current_index];
}

std::vector<float> SupervisedImprintingEnv::get_y() {
  std::vector<float> target;
  target.push_back(this->targets[this->current_index]);
//  target.push_back(10.0);
//  return 10;
  return target;
}

std::vector<float> SupervisedImprintingEnv::create_pattern() {
  std::uniform_int_distribution<int> temp_sampler(0, 29);
  std::vector<float> temp_state;
  for (int i = 0; i < 30; i++)
    temp_state.push_back(0);

  while (sum(temp_state) != 10) {
    int s = temp_sampler(this->mt);
    temp_state[s] = 1;
  }
  return temp_state;
}


