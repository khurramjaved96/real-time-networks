//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/neural_networks/synapse.h"
#include <vector>
#include <iostream>
#include <queue>
#include <mutex>
#include "../../include/neural_networks/neuron.h"
#include "../../include/neural_networks/utils.h"
#include <math.h>

long long int temp=0;
synapse::synapse(neuron *input, neuron *output, float w, float step_size) {
    input_neurons = input;
    output_neurons = output;
    credit = 0;
    weight = w;
    this->step_size = step_size;
    input_neurons->outgoing_synapses.push_back(this);
    output_neurons->incoming_synapses.push_back(this);
    print_status = false;
    float b1=0;
    float b2=0;

}



void synapse::step() {

    output_neurons->value_mutex.lock();
    output_neurons->temp_value +=  input_neurons->value * weight;
    output_neurons->value_mutex.unlock();

    input_neurons->depth_mutex.lock();
    input_neurons->depth = std::max(output_neurons->depth + 1, input_neurons->depth);
    input_neurons->depth_mutex.unlock();

}

//void synapse::update_credit() {
//    this->credit = this->input_neurons->past_activations.front()*this->output_neurons->error_gradient.front();
//}

void synapse::zero_gradient() {
    this->credit = 0;
}

void synapse::update_weight() {
    this->b1 = this->b1*0.9 + this->credit*0.1;
    this->b2 = this->b2*0.99 + (this->credit*this->credit)*0.01;
//    std::cout << this->weight << " " << this->step_size << " " << this->credit << std::endl;
    this->weight += (this->step_size*this->credit)/(sqrt(this->b2) + 0.00000001);
//    std::cout << this->credit << " " << this->weight << std::endl;
//    this->weight += this->step_size*this->credit;

    if(this->weight > 5){
        std::cout << this->credit << " " << this->weight << std::endl;
        std::cout << this->input_neurons->id << " " << this->output_neurons->id << " Cutting weight\n";
        exit(1);
//        this->weight = 2;
    }
    else if(this->weight< -5)
    {std::cout << this->credit << " " << this->weight << std::endl;
        std::cout << this->input_neurons->id << " " << this->output_neurons->id << " Cutting weight\n";
        std::cout << "Cutting weight\n";
        exit(1);
//        this->weight = -2;
    }
}

void synapse::read_gradient() {
    auto grad = this->output_neurons->error_gradient.front();

}