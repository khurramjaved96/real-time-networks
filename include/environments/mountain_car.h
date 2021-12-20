//
// Created by taodav on 29/6/21.
//

#ifndef INCLUDE_ENVIRONMENTS_MOUNTAIN_CAR_H_
#define INCLUDE_ENVIRONMENTS_MOUNTAIN_CAR_H_

#include "./tmaze.h"

class MountainCar {
 protected:
  Observation current_obs;
  std::mt19937 mt;
  std::uniform_int_distribution<int> action_sampler;
  std::uniform_real_distribution<float> state_sampler;

 public:
  float max_position;
  float min_position;
  float max_velocity;
  float min_velocity;
  float goal_position;
  int discretization;
  explicit MountainCar(int seed, int discretization = 0);
  int get_random_action();
  int observation_shape();
  int n_actions();
  bool at_goal();
  Observation get_current_obs();
  Observation reset();
  Observation step(int action);
};


class SparseMountainCar : public MountainCar {
 public:
  explicit SparseMountainCar(int seed, int discretization = 0);
  Observation step(int action);
};

class NonEpisodicMountainCar : public SparseMountainCar{
 public:
  explicit NonEpisodicMountainCar (int seed, int discretization = 0);
  Observation step(int action);
};

#endif  // INCLUDE_ENVIRONMENTS_MOUNTAIN_CAR_H_
