//
// Created by haseebs on 5/2/21.
//
#include <iostream>
#include <vector>
#include <queue>
#include <random>

#ifndef INCLUDE_ENVIRONMENTS_COPY_TASK_H_
#define INCLUDE_ENVIRONMENTS_COPY_TASK_H_

class CopyTask {
  int L;
  int fix_L_val;
  int seq_length;
  int seq_timestep;       // counts timestep in current seq
  int data_timestep;      // counts only target tokens
  int sequence_gap;       // how much gap to put between sequences
  int sequence_gap_left;  // used to count the gap indices when using them
  float total_err_per_seq;
  float decayed_avg_err;
  bool randomize_sequence_length;
  std::mt19937 mt;
  std::vector<float> current_state;
  std::queue<std::vector<float>> past_states;
  std::uniform_int_distribution<int> bit_sampler;

 public:
  CopyTask(int seed, int fix_L_val, bool randomize_sequence_length, int sequence_gap);
  int get_data_timestep();
  int get_L();
  int get_seq_length();
  float get_target();
  std::vector<float> step(float err_last_step);
  std::vector<float> reset();
  std::vector<float> get_state();
};
#endif  // INCLUDE_ENVIRONMENTS_COPY_TASK_H_
