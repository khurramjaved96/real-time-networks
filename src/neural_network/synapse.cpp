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

long long int temp=0;
synapse::synapse(neuron *input, neuron *output, float w) {
    input_neurons = input;
    output_neurons = output;
    credit = 0;
    weight = w;
    step_size = 0.0001;
    input_neurons->outgoing_synapses.push_back(this);
    output_neurons->incoming_synapses.push_back(this);
    print_status = false;
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
    this->weight -= this->step_size*this->credit;
}

void synapse::read_gradient() {
    auto grad = this->output_neurons->error_gradient.front();

}