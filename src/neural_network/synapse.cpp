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
    this->prediction_synapse = false;
    this->b1 = 0;
    this->b2 = 0;
    this->memory_made = false;
}


void synapse::zero_gradient() {
    this->credit = 0;
}

void synapse::update_weight()
{
    this->weight += (this->step_size * this->credit);
//            this->credit = 0;
//    if(this->credit != 0) {
////        this->b1 = this->b1 * 0.9 + this->credit * 0.1;
//        this->b2 = this->b2 * 0.99 + (this->credit * this->credit) * 0.01;
//        this->weight += (this->step_size * this->credit) / (sqrt(this->b2) + 1e-8);
//    }
}

void synapse::read_gradient() {
    auto grad = this->output_neurons->error_gradient.front();

}


no_grad_synapse::no_grad_synapse(neuron *input, neuron *output) {
    this->input_neurons = input;
    this->output_neurons = output;

}

void no_grad_synapse::copy_activation() {
//    if(this->input_neurons->value > 0)
//        std::cout << "Copying val " << this->input_neurons->value << " to " << this->output_neurons->temp_value << std::endl;
//    std::cout << "No grad synapse value  =" <<
    this->output_neurons->temp_value = this->input_neurons->value;
}
