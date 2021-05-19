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
#include <cmath>


neuron::neuron(bool activation) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    neuron_id++;
    useless_neuron = false;
    this->average_activation = 0;
    this->output_neuron = false;
    this->activation_type = activation;
    input_neuron = false;
    memory_made = 0;

}

neuron::neuron(bool activation, bool output_n) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->output_neuron = output_n;
    this->activation_type = activation;
    input_neuron = false;
    memory_made = 0;

}

neuron::neuron(bool activation, bool output_n, bool input_n) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->output_neuron = false;
    this->activation_type = true;
    input_neuron = input_n;
    memory_made = 0;

}


void neuron::init_incoming_synapses() {
    if (this->incoming_synapses.size() > 0) {
        // goal here is to make w1x1 + w2x2 == target_activation_val (which is N(1,0.1))
        float target_activation_val = normal_dist.get_random_number();
        for (auto &it : this->incoming_synapses) {
            it->weight = target_activation_val / (this->incoming_synapses.size() * it->input_neuron->value);
            //std::cout << "id: " << id << " w: " << it->weight << std::endl;
        }
        this->update_value();
        this->fire(0);
    }
}

void neuron::update_value() {
    if(memory_made>0)
        memory_made--;
    float temp_val = 0;
    for (auto &it : this->incoming_synapses) {
        it->age++;
        temp_val += it->weight * it->input_neuron->value;
    }
    this->temp_value = temp_val;
}


bool to_delete_ss(synapse* s)
{
    return s->useless;
}


void neuron::mark_useless_weights(){

    for(auto &it : this->outgoing_synapses){
        if(it->age > 10000){
            if( this->average_activation * std::abs(it->weight) < 0.01)
            {
//                Delete this synapses
                it->useless = true;
//                std::cout << "Deleting useless synapse\n";

            }
            else if(it->output_neuron->useless_neuron)
            {
                // Delete this synapses
//                std::cout << "Deleting useless synapse\n";
//                std::cout << "Shouldn't get here\n";
//                exit(1);
                it->useless = true;
            }
        }
    }


    if(this->outgoing_synapses.empty() and !this->output_neuron)
    {
//        std::cout << "Deleting useless neuron\n";
        this->useless_neuron = true;
        for(auto it : this->incoming_synapses)
            it->useless = true;
    }
    if(this->input_neuron)
        this->useless_neuron = false;
}

void neuron::prune_useless_weights(){
    auto it = std::remove_if(this->outgoing_synapses.begin(), this->outgoing_synapses.end(), to_delete_ss);
    this->outgoing_synapses.erase(it, this->outgoing_synapses.end());

    it = std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete_ss);
    this->incoming_synapses.erase(it, this->incoming_synapses.end());
    if(this->outgoing_synapses.size() <2)
        this->memory_made = false;
}

//void neuron::prune_useless_weights(){
//    std::remove_if(this->outgoing_synapses.begin(), this->outgoing_synapses.end(), to_delete);
//    std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete);
//}

void neuron::fire(int time_step) {
    this->value = temp_value;
    if (this->activation_type && this->value <= 0)
    {
        this->value = 0;
    } else {
        this->average_activation = this->average_activation * 0.95 + 0.05 * std::abs(this->value);
    }
    temp_value = 0;
    this->past_activations.push(std::pair<float, int>(this->value, time_step));
    //std::cout << "n: " << id << "avg_activation_val: " << this->value << std::endl;
}

void neuron::forward_gradients() {

    if (!this->error_gradient.empty()) {
        for (auto &it : this->incoming_synapses) {
            if(it->pass_gradients) {
                float message_value;

                message_value = this->error_gradient.front().gradient;

                message grad_temp(message_value, this->error_gradient.front().time_step);
                grad_temp.lambda = this->error_gradient.front().lambda;
                grad_temp.gamma = this->error_gradient.front().gamma;
                grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
                it->grad_queue.push(grad_temp);
            }

        }
        this->error_gradient.pop();
    }
}

float neuron::introduce_targets(float target, int time_step) {
    if (!this->past_activations.empty()) {
        float error = target - this->past_activations.front().first;
        float error_grad = error;
//        If activation was zero, we applied relu. Gradients don't flow back
        if (this->past_activations.front().first <= 0 and this->activation_type) {
            error_grad = 0;
        }
//        std::cout << "Error = " << error << std::endl;
        message m(error_grad, time_step);
        this->error_gradient.push(m);
        this->past_activations.pop();
        return error * error;
    }
    return 0;
}

float neuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
    if (!this->past_activations.empty()) {
        float error = target - this->past_activations.front().first;
        float error_grad = error;
//        If activation was zero, we applied relu. Gradients don't flow back
        if (this->past_activations.front().first <= 0 and this->activation_type) {
            std::cout << "Should never get here\n";
            exit(1);
            error_grad = 0;
        }
//        std::cout << "Error = " << error << std::endl;
        message m(error_grad, time_step);
        m.lambda = lambda;
        m.gamma = gamma;
        this->error_gradient.push(m);
        this->past_activations.pop();
        return error * error;
    }
    return 0;
}


void neuron::propogate_error() {

    float accumulate_gradient = 0;
    std::vector<int> time_vector;
    std::vector<int> distance_vector;
    std::vector<int> activation_time_required_list;
    std::vector<int> queue_len_vector;
    int time_check = 99999;

    if (!this->incoming_synapses.empty() or
        true) { //Only compute gradient for the hidden node if further propagation is required
        if (!this->outgoing_synapses.empty()) { // No gradient computation required for prediction nodes
            bool flag = false;
            bool wait = false;
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
                            wait = true;
//                            std::cout << "Shouldn't happen in normal operation. Implementation deferred for later\n";
//                            exit(1);
                        }
                        if (!wait) {
                            time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
                            distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);
                            queue_len_vector.push_back(output_synapses_iterator->grad_queue.size());
//                    Only accumulate gradient if activation was non-zero.
                            if (this->past_activations.front().first > 0 or !this->activation_type) {
//                            std::cout << "Past activation = " << this->past_activations.front().first << std::endl;
                                accumulate_gradient += output_synapses_iterator->weight *
                                                       output_synapses_iterator->grad_queue.front().gradient;
                            }


                            if (time_check == 99999) {
                                time_check = activation_time_required;
                            } else {
                                if (time_check != activation_time_required) {

                                    flag = true;
                                }
                            }
                        }
                    }
                } else {
//                    flag = true;
                }

            }


            if (flag or time_vector.empty())
                return;
            for (auto &it : this->outgoing_synapses) {
                if (it->grad_queue.size() > 0 and !wait) {
                    if (it->output_neuron->output_neuron) {
//                        if(it->trace > 0 or it->grad_queue.front().gradient > 0){
//                            std::cout << it->trace << "\t" << it->grad_queue.front().gradient << std::endl;
//                        }
                        it->trace = it->trace * it->grad_queue.front().gamma * it->grad_queue.front().lambda + this->past_activations.front().first;
//                        if(it->trace > 0)
//                        std::cout << it->trace  <<std::endl;
//                        if(this->id == 1 and it->trace > 0)
//                        std::cout << it->trace << std::endl;
                        it->credit = it->grad_queue.front().gradient * it->trace;

                        it->credit_activation_idbd = this->past_activations.front().first;
                        it->grad_queue.pop();
                    } else {

                        it->credit = it->grad_queue.front().gradient *
                                     this->past_activations.front().first;
                        it->credit_activation_idbd = this->past_activations.front().first;
                        it->grad_queue.pop();
                    }
                } else {
                    it->credit = 0;
                }
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
normal_random neuron::normal_dist = normal_random(0, 1, 0.1);