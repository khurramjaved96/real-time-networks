//
// Created by haseebs on 5/2/21.
//
#include <iostream>
#include <vector>
#include <queue>
#include <random>

#ifndef FLEXIBLENN_COPY_TASK_H
#define FLEXIBLENN_COPY_TASK_H

class CopyTask{
    int L;
    int seq_length;
    int seq_timestep;     // counts timestep in current seq
    int data_timestep;    // counts only target tokens
    int sequence_gap_left;
    float total_err_per_seq;
    float decayed_avg_err;
    bool randomize_sequence_length;
    std::mt19937 mt;
    std::vector<float> current_state;
    std::queue<std::vector<float>> past_states;
    std::uniform_int_distribution<int> bit_sampler;

public:
    CopyTask(int seed, bool randomize_sequence_length);
    int get_data_timestep();
    int get_L();
    int get_seq_length();
    float get_target();
    std::vector<float> step(float err_last_step);
    std::vector<float> reset();
    std::vector<float> get_state();

};

#endif //FLEXIBLENN_COPY_TASK_H
