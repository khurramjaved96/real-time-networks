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


neuron::neuron() {
    value = 0;
    depth = 1;
    temp_value = 0;
    id = neuron_id;
    neuron_id++;
}


void neuron::activation() {
    if (this->temp_value > 0) this->value = this->temp_value;
    else this->value = 0;

    this->temp_value = 0;
//    this->past_activations.push(this->value);
}

void neuron::update_value() {

    float temp_val = 0;
    for (auto &it : this->incoming_synapses) {
        temp_val += it->weight * it->input_neurons->value;
    }
    this->temp_value = temp_val;
}

void neuron::fire(int time_step) {
    if (this->temp_value > 0) {
        this->value = temp_value;
    } else {
        this->value = 0;
    }
    this->past_activations.push(std::pair<float, int>(this->value, time_step));
    temp_value = 0;
}

void neuron::forward_gradients() {

    if (!this->error_gradient.empty()) {
        for (auto &it : this->incoming_synapses) {
//            std::cout << "Pushing gradient to synapses for Neuron ID " << this->id << " " << " To Neuron "
//                      << it->input_neurons->id
//                      << std::endl;
//            std::cout << "Forward gradient time-step = " << this->error_gradient.front().time_step << std::endl;
            message grad_temp(this->error_gradient.front().message_value, this->error_gradient.front().time_step);
            grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
            it->grad_queue.push(grad_temp);
        }
        this->error_gradient.pop();
    }
}

void neuron::introduce_targets(float target, int time_step) {
//    std::cout << "Adding targets for time step " << time_step << "\n";
    if (!this->past_activations.empty()) {

        assert(time_step == this->past_activations.front().second);
        message m(target - this->past_activations.front().first, time_step);
//        std::cout << "Target error = " << target << " - " << this->past_activations.front().first << " " << "time step "
//                  << time_step << std::endl;
//        std::cout << "Depth = \t" << m.distance_travelled << std::endl;
        this->error_gradient.push(m);
        this->past_activations.pop();
    }
}

//
void neuron::propogate_error() {
//    std::cout << "Gradient propagation from synapses to neuron " << this->id << "\n";
    float sum_gradient = 0;
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
                        std::cout << "Deleting activation from " << this->id << std::endl;
                        output_synapses_iterator->grad_queue.pop();
                    }

                    if (output_synapses_iterator->grad_queue.empty()) {
                        std::cout << "Waiting for gradient from other paths; skipping propagation =\n";
                        flag = true;
//                        return;
                    }
                    if(!flag) {
                        activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
                                                   output_synapses_iterator->grad_queue.front().distance_travelled - 1;
                        activation_time_required_list.push_back(activation_time_required);
                        if (this->past_activations.front().second < activation_time_required) {
                            std::cout << "Activation time = " << this->past_activations.front().second
                                      << " Time required "
                                      << activation_time_required << std::endl;
//                        std::cout << output_synapses_iterator->grad_queue.front().time_step << " "
//                                  << output_synapses_iterator->grad_queue.front().distance_travelled << std::endl;
//                        std::cout << this->past_activations.front().second << " " << activation_time_required
//                                  << std::endl;
                            std::cout << "Output synapses time = "
                                      << output_synapses_iterator->grad_queue.front().time_step
                                      << " Distance travelled = "
                                      << output_synapses_iterator->grad_queue.front().distance_travelled << std::endl;
                            std::cout << "Shouldn't happen in normal operation. Implementation deferred for later\n";
                            exit(1);
                        }

                        time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
                        distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);

                        output_synapses_iterator->credit = output_synapses_iterator->grad_queue.front().message_value *
                                                           this->past_activations.front().second;

//                    Only accumulate gradient if activation was non-zero.
                        if (this->past_activations.front().second > 0)
                            sum_gradient += output_synapses_iterator->weight *
                                            output_synapses_iterator->grad_queue.front().message_value;

                        if (time_check == 99999) {
                            time_check = activation_time_required;
                        } else {
                            if (time_check != activation_time_required) {
                                flag = true;
                            }
                        }
                    }
                } else {
                    std::cout << "Waiting for gradient from other paths; skipping propagation\n";
                    flag = true;
                }

//                output_synapses_iterator->output_neurons->error_gradient.pop();
            }
            if (flag)
                return;
            for (auto &it : this->outgoing_synapses) {
                it->credit = it->grad_queue.front().message_value *
                             this->past_activations.front().second;
                it->grad_queue.pop();
            }
//            if(this->incoming_synapses.empty())
//            {
//                this->past_activations.pop();
//                return;
//            }
            std::cout << "Gradient propagation successful \n";
//            std::cout << "Time check for new message " << time_check << " " << time_vector.size() <<  std::endl;

            message n_message(sum_gradient, time_vector[0]);


            auto it = std::max_element(distance_vector.begin(), distance_vector.end());
            n_message.distance_travelled = *it;
            print_vector(distance_vector);
            print_vector(time_vector);
            print_vector(activation_time_required_list);
            std::cout << "Time of activation " << this->past_activations.front().second << std::endl;
//            print_vector(sum(distance_vector, time_vector));
            std::cout << "Message distance travelled = " << n_message.distance_travelled << " Message time step"
                      << n_message.time_step << std::endl;
            std::cout << "Oldest activaton time = " << this->past_activations.front().second << std::endl;
            this->past_activations.pop();
            this->error_gradient.push(n_message);
        } else { std::cout << "No outgoing nodes for current neuron with id " << this->id << std::endl; }
    } else {
        std::cout << "No incoming nodes for current neuron with id " << this->id << std::endl;
        std::cout << "Implement gradient for weigth parameter here \n";
        if (!this->outgoing_synapses.empty()) {
            for (auto &output_synapses_iterator : this->outgoing_synapses) {
                if (!output_synapses_iterator->grad_queue.empty()) {
                    output_synapses_iterator->grad_queue.pop();
                }
            }
        }
    }
}

int neuron::neuron_id = 0;