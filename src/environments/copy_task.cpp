//
// Created by haseebs on 5/2/21.
//

#include "../../include/environments/copy_task.h"
#include <vector>
#include <math.h>
#include <random>

CopyTask::CopyTask(int seed, int fix_L_val, bool randomize_sequence_length, int sequence_gap) : mt(seed) {
  this->L = 1;
  this->fix_L_val = fix_L_val;
  if (this->fix_L_val)
    this->L = this->fix_L_val;
  this->seq_length = 1;
  this->seq_timestep = 0;
  this->data_timestep = 0;
  this->total_err_per_seq = 0;
  this->decayed_avg_err = 1;
  this->sequence_gap_left = 0;
  this->sequence_gap = sequence_gap;
  this->randomize_sequence_length = randomize_sequence_length;
  this->bit_sampler = std::uniform_int_distribution<int>(0, 1);
}

int CopyTask::get_data_timestep() {
  return this->data_timestep;
}

int CopyTask::get_L() {
  return this->L;
}

int CopyTask::get_seq_length() {
  return this->seq_length;
}

std::vector<float> CopyTask::get_state() {
  // state: [start_of_stimuli_flag, end_of_stimuli_flag, stimuli]
  return this->current_state;
}

float CopyTask::get_target() {
  //TODO this shouldnt be called before step
  // target is 0 until we've passed the pred token
  float target = 0;
  if (this->seq_timestep > this->seq_length + 1) {
    target = past_states.front()[2];
    past_states.pop();
  }
  return target;
}

std::vector<float> CopyTask::reset() {
  this->L = 1;
  if (this->fix_L_val)
    this->L = this->fix_L_val;
  this->seq_length = 1;
  this->seq_timestep = 0;
  this->data_timestep = 0;
  this->total_err_per_seq = 0;
  this->decayed_avg_err = 1;
  this->current_state = std::vector < float > {1, 0, 0};
  return this->current_state;
}

std::vector<float> CopyTask::step(float err_last_step) {
  // the gap between sequence
  if (this->sequence_gap_left > 0) {
    this->sequence_gap_left -= 1;
    this->current_state = std::vector < float > {0, 0, 0};
    if (this->sequence_gap_left == 0)
      this->current_state = std::vector < float > {1, 0, 0};
    return this->current_state;
  }

  this->total_err_per_seq += abs(err_last_step);
  // after obtaining err for first pred
  //if(this->seq_timestep > this->seq_length+1)

  // after the seq + flag + pred is over and new seq started
  if (this->seq_timestep > this->seq_length * 2) {
    float err_per_bit = this->total_err_per_seq / this->seq_length;
    this->decayed_avg_err = (this->decayed_avg_err * 0.9) + (err_per_bit * 0.1);
    //std::cout << "avg err: " << this->decayed_avg_err << std::endl;
    if (this->decayed_avg_err < 0.10 && !this->fix_L_val) {
      this->L += 1;
      this->decayed_avg_err = 1;
    }
    this->seq_length = this->L;
    if (this->randomize_sequence_length) {
      auto seq_len_sampler = std::uniform_int_distribution<int>(std::max(this->L - 5, 1), this->L);
      this->seq_length = seq_len_sampler(mt);
    }
    this->seq_timestep = 0;
    this->total_err_per_seq = 0;
    this->sequence_gap_left = this->sequence_gap;
    this->current_state = std::vector < float > {0, 0, 0};
    return current_state;
  }

  if (this->seq_timestep < this->seq_length) {
    this->current_state = std::vector < float > {0, 0, float(bit_sampler(mt))};
    this->past_states.push(this->current_state);
  }
    // throw a pred flag
  else if (this->seq_timestep == this->seq_length)
    this->current_state = std::vector < float > {0, 1, 0};
    // start the pred sequence
  else if (this->seq_timestep > this->seq_length) {
    this->data_timestep += 1;
    this->current_state = std::vector < float > {0, 0, 0};
  }

  this->seq_timestep += 1;
  return current_state;
}
