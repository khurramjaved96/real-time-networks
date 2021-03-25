//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/neural_networks/synapse.h"
#include <vector>
#include <iostream>
#include <queue>
#include <mutex>


long long int temp=0;
synapse::synapse(neuron *input, neuron *output, float w) {
    input_neurons = input;
    output_neurons = output;
    weight = w;
}

void synapse::step() {
//    std::lock_guard<std::mutex> guard(global);
    output_neurons->value_mutex.lock();
    output_neurons->temp_value +=  input_neurons->value * weight;
    output_neurons->value_mutex.unlock();

    input_neurons->depth_mutex.lock();
    input_neurons->depth = std::max(output_neurons->depth + 1, input_neurons->depth);
    input_neurons->depth_mutex.unlock();

}

//void synapse::sync() {
//    if (output_neurons->temp_value>0) {
//        output_neurons->value = output_neurons->temp_value;
//    }
//    output_neurons->temp_value=0;
////    output_neurons->past_activations.push();
//
//}
