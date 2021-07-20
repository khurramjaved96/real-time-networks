//
// Created by Khurram Javed on 2021-04-22.
//

#include "../../include/animal_learning/tracecondioning.h"
#include <vector>
#include <math.h>
#include <random>
#include "../../include/utils.h"

TracePatterning::TracePatterning(std::pair<int, int> ISI, std::pair<int, int> ISI_long, std::pair<int, int> ITI,
                                 int num_distractors, int seed) : ISI_sampler(ISI.first, ISI.second),
                                                                  ISI_long_sampler(ISI_long.first, ISI_long.second),
                                                                  ITI_sampler(ITI.first, ITI.second), mt(seed),
                                                                  NoiseSampler(0, 1) {
    this->num_distractors = num_distractors;
    this->pattern_len = 6;
    for (int temp = 0; temp < this->pattern_len + this->num_distractors + 1; temp++) {
        current_state.push_back(0);
    }

    requires_reset = true;
    remaining_steps = 0;
    remaining_until_US = 0;

    while (this->valid_patterns.size() < 10) {
        std::vector<float> p = create_pattern();
        bool flag = true;
        for (auto it: this->valid_patterns) {
            if (p == it) {
                flag = false;
            }
        }
        if (flag) {
            this->valid_patterns.push_back(p);
        }
    }

}


std::vector<float> TracePatterning::get_state() {
    return this->current_state;
}

std::vector<float> TracePatterning::create_pattern() {
    std::uniform_int_distribution<int> temp_sampler(0, 5);
    std::vector<float> temp_state{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    while (sum(temp_state) != 3) {
        int s = temp_sampler(this->mt);
        temp_state[s] = 1;
    }
    return temp_state;
}


std::vector<float> TracePatterning::step() {
    for (int a = 0; a < this->pattern_len + this->num_distractors; a++)
        this->current_state[a] = 0;

    if (remaining_steps == 0) {

        return this->reset();
    }
    set_noise_bits();
    if (this->remaining_until_US == 1 and this->valid) {
        this->current_state[6] = 1;
    } else {
        this->current_state[6] = 0;
    }


    this->remaining_until_US--;
    this->remaining_steps--;
    return current_state;
}

std::vector<float> TracePatterning::reset() {
    this->remaining_until_US = ISI_sampler(mt);
    this->remaining_until_US_long = ISI_long_sampler(mt);
    this->remaining_steps = this->remaining_until_US_long + ITI_sampler(mt);
    std::vector<float> state_pattern = this->create_pattern();
    for (int a = 0; a < 6; a++)
        this->current_state[a] = state_pattern[a];
    this->valid = false;
    for (auto it : this->valid_patterns) {
        if (it == state_pattern) {
            this->valid = true;
            break;
        }
    }
    this->set_noise_bits();
    return current_state;
}

void TracePatterning::set_noise_bits() {
    for (int temp = this->pattern_len + 1; temp < this->pattern_len + 1 + this->num_distractors; temp++) {

        if (NoiseSampler(mt) > 0.98 or temp == this->pattern_len + 1)
//        if(temp == this->pattern_len + 1)
        {
            this->current_state[temp] = 1;
        } else {
            this->current_state[temp] = 0;
        }
    }
}

float TracePatterning::get_US() {
    return this->current_state[6];
}


float TracePatterning::get_target(float gamma) {
    if (this->remaining_until_US >= 0 and this->valid) {
        return pow(gamma, this->remaining_until_US);
    }
    return 0;
}



