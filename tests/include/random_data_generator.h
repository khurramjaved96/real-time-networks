//
// Created by Khurram Javed on 2021-08-11.
//

#ifndef TESTS_INCLUDE_RANDOM_DATA_GENERATOR_H_
#define TESTS_INCLUDE_RANDOM_DATA_GENERATOR_H_

#include <vector>

class RandomDataGenerator{
 public:
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> targets;
  int current_position = 0;
  std::vector<float> get_input();
  std::vector<float> get_target();
  bool step();
  RandomDataGenerator();
};
#endif //TESTS_INCLUDE_RANDOM_DATA_GENERATOR_H_
