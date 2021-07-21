//
// Created by taodav on 29/6/21.
//

#ifndef FLEXIBLENN_MOUNTAIN_CAR_H
#define FLEXIBLENN_MOUNTAIN_CAR_H

#include "tmaze.h"


class MountainCar {
    Observation current_obs;
    std::mt19937 mt;
    std::uniform_int_distribution<int> action_sampler;

public:
    float max_position;
    float min_position;
    float max_velocity;
    float min_velocity;
    float goal_position;
    int discretization;
    int max_timesteps;
    MountainCar(int seed, int discretization = 0);
    int get_random_action();
    unsigned long observation_shape();
    int n_actions();
    bool at_goal();
    Observation get_current_obs();
    Observation reset();
    Observation step(int action);
};


#endif //FLEXIBLENN_MOUNTAIN_CAR_H
