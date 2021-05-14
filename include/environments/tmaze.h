#include <iostream>
#include <vector>
#include <queue>
#include <random>

#ifndef FLEXIBLENN_COPY_TASK_H
#define FLEXIBLENN_COPY_TASK_H

struct Observation
{
    int timestep;
    int episode;
    float reward;
    bool is_terminal;
    std::vector<float> state;
};


class TMaze{
    //obs.state: [110]/[011] at start, [101] in corridor, [010] at junction
    //actions: N[1000], S[0001], E[0100], W[0010]
    //corridor goes from E->W
    //correct directions are state=[110]:N and state=[011]:S
    std::mt19937 mt;
    int length_of_corridor;
    int current_pos_in_corridor;
    int current_episode;
    Observation current_obs;
    std::vector<float> direction_state;
    std::vector<float> correct_direction; //[1000] or [0001]
    std::uniform_int_distribution<int> direction_sampler;
    std::uniform_int_distribution<int> action_sampler;
    std::vector<float> generate_direction_state();

    const std::vector<float> N = {1,0,0,0};
    const std::vector<float> E = {0,1,0,0};
    const std::vector<float> W = {0,0,1,0};
    const std::vector<float> S = {0,0,0,1};

    const std::vector<float> corridor_state = {1,0,1};
    const std::vector<float> junction_state = {0,1,0};
    const std::vector<float> terminal_state = {0,0,0};

public:
    TMaze(int seed, int length_of_corridor);
    int get_current_pos_in_corridor();
    int get_length_of_corridor();
    void set_length_of_corrider(int value);
    std::vector<float> get_random_action();
    Observation get_current_obs();
    Observation reset();
    Observation step(std::vector<float> action);

};

#endif //FLEXIBLENN_COPY_TASK_H
