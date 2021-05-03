//
// Created by haseebs on 5/2/21.
//

#include "../../include/environments/copy_task.h"
#include <vector>
#include <math.h>
#include <random>

CopyTask::CopyTask(int seed): mt(seed) {
    this->seq_length = 1;
    this->seq_timestep = 0;
    this->data_timestep = 0;
    this->current_timestep = 0;
    this->total_err_per_seq = 0;
    this->bit_sampler = std::uniform_int_distribution<int>(0,1);
}


int CopyTask::get_datatime(){
    return this->data_timestep;
}


std::vector<float> CopyTask::get_state(){
    return this->current_state;
}


float CopyTask::get_target(){
    //TODO this shouldnt be called before step
    // target is 0 until we've passed the pred token
    float target = 0;
    if(this->seq_timestep > this->seq_length + 1){
        target = past_states.front()[1];
        past_states.pop();
    }
    return target;
}


std::vector<float> CopyTask::step(float err_last_step){
    // after obtaining err for first pred
    if(this->seq_timestep > this->seq_length+1)
        this->total_err_per_seq += err_last_step;

    // after the seq + flag + pred is over and new seq started
    if(this->seq_timestep > this->seq_length*2){
        float err_per_bit = this->total_err_per_seq / this->seq_length;
        if(err_per_bit < 0.15)
            this->seq_length += 1;
        this->seq_timestep = 0;
        this->total_err_per_seq = 0;
    }

    if(this->seq_timestep < this->seq_length){
        this->current_state = std::vector<float>{0, float(bit_sampler(mt))};
        this->past_states.push(this->current_state);
    }
    // throw a pred flag
    else if(this->seq_timestep == this->seq_length)
        this->current_state = std::vector<float>{1, 0};
    // start the pred sequence
    else if(this->seq_timestep > this->seq_length){
        this->data_timestep += 1;
        this->current_state = std::vector<float>{0, 0};
    }

    this->seq_timestep += 1;
    this->current_timestep += 1;
    return current_state;
}
