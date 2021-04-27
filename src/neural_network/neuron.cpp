//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/neural_networks/neuron.h"
#include <iostream>
#include "../../include/neural_networks/message.h"
#include <utility>
#include <cassert>
#include <algorithm>
#include <vector>
#include "../../include/neural_networks/utils.h"
#include "../../include/utils.h"
#include <assert.h>
#include <math.h>


neuron::neuron(bool activation) {
    value = 0;
    depth = 1;
    temp_value = 0;
    id = neuron_id;
    neuron_id++;
    this->activation_type = activation;

}

void neuron::update_value() {

    float temp_val = 0;
    for (auto &it : this->incoming_synapses) {
        temp_val += it->weight * it->input_neurons->value;
    }
    this->temp_value = temp_val;
}

void neuron::fire(int time_step) {
    this->value = temp_value;
    if (this->activation_type && this->value <= 0) {
        this->value = 0;
    }
    temp_value = 0;
    this->past_activations.push(std::pair<float, int>(this->value, time_step));

}

void neuron::forward_gradients() {

    if (!this->error_gradient.empty()) {
        for (auto &it : this->incoming_synapses) {
            float message_value = 0;
            if (this->past_activations.front().first > 0 or !activation_type or true) {
                message_value = this->error_gradient.front().message_value;
            }
            else{
                message_value = 0;
            }
            message grad_temp(message_value, this->error_gradient.front().time_step);
            grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
            it->grad_queue.push(grad_temp);

        }
        this->error_gradient.pop();
    }
}

float neuron::introduce_targets(float target, int time_step) {
    if (!this->past_activations.empty()) {

        assert(time_step == this->past_activations.front().second);
        float error = target - this->past_activations.front().first;
        float error_grad = error ;
        if(this->past_activations.front().first <= 0 and this->activation_type){
            error_grad = 0;
        }
//        std::cout << "Error = " << error << std::endl;
        message m(error_grad, time_step);
        this->error_gradient.push(m);
        this->past_activations.pop();
        return error*error;
    }
    return 0;
}


void neuron::propogate_error() {
    float accumulate_gradient = 0;
    std::vector<int> time_vector;
    std::vector<int> distance_vector;
    std::vector<int> activation_time_required_list;
    int time_check = 99999;

    if (!this->incoming_synapses.empty() or
        true) { //Only compute gradient for the hidden node if further propagation is required
        if (!this->outgoing_synapses.empty()) { // No gradient computation required for prediction nodes
            bool flag = false;
            for (auto &output_synapses_iterator : this->outgoing_synapses) { //Iterate over all outgoing synapses. We want to make sure

                if (!output_synapses_iterator->grad_queue.empty()) {

                    int activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
                                                   output_synapses_iterator->grad_queue.front().distance_travelled - 1;
                    while (!output_synapses_iterator->grad_queue.empty() and this->past_activations.front().second >
                                                                             activation_time_required) {
//                        Activation for this gradient is not stored; this can happen in the beginning of the network initalization, or if new paths are introduced at run time. We just drop this gradient.
                        output_synapses_iterator->grad_queue.pop();
                    }

                    if (output_synapses_iterator->grad_queue.empty()) {
//                        "Waiting for gradient from other paths; skipping propagation"
                        flag = true;
                    }
                    if (!flag) {
                        assert(!output_synapses_iterator->grad_queue.empty());
                        activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
                                                   output_synapses_iterator->grad_queue.front().distance_travelled - 1;
                        activation_time_required_list.push_back(activation_time_required);
                        if (this->past_activations.front().second < activation_time_required) {
                            std::cout << "Shouldn't happen in normal operation. Implementation deferred for later\n";
                            exit(1);
                        }

                        time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
                        distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);

//                    Only accumulate gradient if activation was non-zero.
                        if (this->past_activations.front().first > 0 or !this->activation_type) {
//                            std::cout << "Past activation = " << this->past_activations.front().first << std::endl;
                        accumulate_gradient += output_synapses_iterator->weight *
                                               output_synapses_iterator->grad_queue.front().message_value;
                        }

                        if (time_check == 99999) {
                            time_check = activation_time_required;
                        } else {
                            if (time_check != activation_time_required) {
                                flag = true;
                            }
                        }
                    }
                } else {
                    flag = true;
                }

            }
            if (flag)
                return;
            for (auto &it : this->outgoing_synapses) {
                it->credit = it->grad_queue.front().message_value *
                             this->past_activations.front().first;
                it->grad_queue.pop();
            }


            message n_message(accumulate_gradient, time_vector[0]);
            auto it = std::max_element(distance_vector.begin(), distance_vector.end());
            n_message.distance_travelled = *it;
            this->past_activations.pop();
            this->error_gradient.push(n_message);

        }
    }
}

int neuron::neuron_id = 0;