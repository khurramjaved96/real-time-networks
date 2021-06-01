//
// Created by Khurram Javed on 2021-04-22.
//

#include "../../include/animal_learning/tracecondioning.h"
#include <vector>
#include <math.h>
#include <random>
#include "../../include/utils.h"

TracePatterning::TracePatterning(std::pair<int, int> ISI, std::pair<int, int> ISI_long,  std::pair<int, int> ITI, int num_distractors, int seed): ISI_sampler(ISI.first, ISI.second), ISI_long_sampler(ISI_long.first, ISI_long.second), ITI_sampler(ITI.first, ITI.second), mt(seed), NoiseSampler(0, 1) {
    this->num_distractors = num_distractors;
    for(int temp = 0; temp< 8; temp++)
    {
        current_state.push_back(0);
    }
    this->pattern_len = 6;
    requires_reset = true;
    remaining_steps = 0;
    remaining_until_US = 0;
    remaining_until_US_long = 0;
    ISI_length = ISI.first;
    turn_off_first = false;
    while(this->valid_patterns.size() < 10){
        std::vector<float> p = create_pattern();
        bool flag = true;
        for(auto it: this->valid_patterns){
            if(p == it){
                flag = false;
            }
        }
        if(flag){
            this->valid_patterns.push_back(p);
        }
    }
    this->distribution.push_back(0);
    this->distribution.push_back(0);
//    std::cout << "Valid patterns selected\n";
//    for(auto it: this->valid_patterns){
//        print_vector(it);
//    }
//    exit(1);
}


void TracePatterning::increase_ISI(int t) {
    ISI_length+= t;
    this->ISI_sampler = std::uniform_int_distribution<int>(ISI_length, ISI_length);
}

std::vector<float> TracePatterning::get_state() {
    return this->current_state;
}

std::vector<float> TracePatterning::create_pattern() {
    std::uniform_int_distribution<int> temp_sampler(0, 5);
    std::vector<float> temp_state{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    while(sum(temp_state) != 3)
    {
        int s =  temp_sampler(this->mt);
        temp_state[s] = 1;
    }
    return temp_state;
}

void TracePatterning::remove_first_US()
{
    this->turn_off_first = true;
}
std::vector<float> TracePatterning::step(){
    for(int a =0; a<8; a++)
        this->current_state[a] = 0;

    if(remaining_steps == 0){

        return this->reset();
    }
    set_noise_bits();
    if(this->remaining_until_US == 1 and this->valid and !this->turn_off_first)
    {
        this->current_state[6] = 1;
    }
    else{
        this->current_state[6] = 0;
    }
    if(this->remaining_until_US_long == 1 and this->valid)
    {
        this->current_state[7] = 0;
    }
    else{
        this->current_state[7] = 0;
    }

    this->remaining_until_US--;
    this->remaining_until_US_long--;
    this->remaining_steps--;
    return current_state;
}

std::vector<float> TracePatterning::reset() {
    this->remaining_until_US = ISI_sampler(mt);
    this->remaining_until_US_long = ISI_long_sampler(mt);
    this->remaining_steps = this->remaining_until_US_long + ITI_sampler(mt);
    std::vector<float> state_pattern = this->create_pattern();
    for(int a = 0; a< 6; a++)
        this->current_state[a] = state_pattern[a];
    this->valid = false;
    for(auto it : this->valid_patterns)
    {
        if(it == state_pattern)
        {
            this->valid = true;
            this->distribution[0]++;
            break;
        }
    }
    if(this->valid == false){
        this->distribution[1]++;
//        std::cout << "False pattern\n";

    }
//    std::cout << "Printing distribution\n\n\n\n";
//    print_vector(this->distribution);
//
//    set_noise_bits();
//    current_state[0] = 1; // Setting the CS
//    current_state[1] = 0; //setting the US
    return current_state;
}

void TracePatterning::set_noise_bits() {
//    for(int temp = 3; temp < this->current_state.size(); temp++)
//    {
//        if(NoiseSampler(mt) > 0.90)
//        {
//            this->current_state[temp] = 1;
//        }
//        else{
//            this->current_state[temp] = 0;
//        }
//    }
}

float TracePatterning::get_US(){
    return this->current_state[6];
}

float TracePatterning::get_long_US(){
    return this->current_state[7];
}

float TracePatterning::get_target(float gamma) {
    if(this->remaining_until_US>=0 and this->valid and !this->turn_off_first)
    {
        return pow(gamma, this->remaining_until_US);
    }
    return 0;
}

float TracePatterning::get_target_long(float gamma) {
    if(this->remaining_until_US_long>=0 and this->valid)
    {
        return pow(gamma, this->remaining_until_US_long);
    }
    return 0;
}//
// Created by Khurram Javed on 2021-05-04.
//

