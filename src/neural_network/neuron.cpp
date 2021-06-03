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
    this->is_output_neuron = false;
    this->activation_type = activation;
    is_input_neuron = false;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;
}

neuron::neuron(bool activation, bool output_n) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->is_output_neuron = output_n;
    this->activation_type = activation;
    is_input_neuron = false;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;
}

neuron::neuron(bool activation, bool output_n, int id) {
    value = 0;
    temp_value = 0;
    this->id = id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->is_output_neuron = output_n;
    this->activation_type = activation;
    is_input_neuron = false;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;
}

neuron::neuron(bool activation, bool output_n, bool input_n) {
    value = 0;
    temp_value = 0;
    id = neuron_id;
    useless_neuron = false;
    this->average_activation = 0;
    neuron_id++;
    this->is_output_neuron = false;
    this->activation_type = true;
    is_input_neuron = input_n;
    memory_made = 0;
    neuron_age = 0;
    mature = false;
    references = 0;

}

void neuron::update_value() {
    if (this->neuron_age == 19999 and !this->is_output_neuron) {
        this->mature = true;
    }
    this->neuron_age++;
    if (memory_made > 0)
        memory_made--;
    this->temp_value = 0;
    if (this->neuron_age == 19999 and !this->is_input_neuron and this->average_activation > 0 and
        this->outgoing_synapses.size() > 0) {
        float scale = 1 / this->average_activation;
//        std::cout << scale << std::endl;
        for (auto it : this->incoming_synapses) {
            it->weight = it->weight * scale;
        }
//
        if (this->outgoing_synapses.size() == 0) {
            std::cout << "Too many outgoing synapses; shouldn't happen\t" << this->outgoing_synapses.size() << "\n";
            std::cout << "ID\t" << this->neuron_id << " Age \t" << this->neuron_age << std::endl;
            exit(1);
        }
        for (auto out_g : this->outgoing_synapses) {
            out_g->weight = out_g->weight * this->average_activation;
            out_g->step_size = 1e-4;
            out_g->turn_on_idbd();
        }
        this->average_activation = 1;
    }

    for (auto &it : this->incoming_synapses) {
        it->age++;
        this->temp_value += it->weight * it->input_neuron->value;
    }
}


bool to_delete_ss(synapse *s) {
    return s->useless;
}


void neuron::mark_useless_weights() {

    for (auto &it : this->outgoing_synapses) {
        if (it->age > 69999) {
            if (!(it->input_neuron->is_input_neuron and it->output_neuron->is_output_neuron)) {
                if (this->average_activation * std::abs(it->weight) < 0.01) {
                    it->useless = true;
                } else if (it->output_neuron->useless_neuron) {
                    it->useless = true;
                }
            }
        }
    }


    if (this->outgoing_synapses.empty() and !this->is_output_neuron and !this->is_input_neuron) {
        this->useless_neuron = true;
        for (auto it : this->incoming_synapses)
            it->useless = true;
    }

    if (this->is_input_neuron)
        this->useless_neuron = false;
}

void neuron::prune_useless_weights() {
    std::for_each(
//            std::execution::seq,
            this->outgoing_synapses.begin(),
            this->outgoing_synapses.end(),
            [&](synapse *s) {
                if(s->useless) {
                    s->decrement_reference();
                    if(s->input_neuron != nullptr) {
                        s->input_neuron->decrement_reference();
                        s->input_neuron = nullptr;
                    }
                    if(s->output_neuron != nullptr) {
                        s->output_neuron->decrement_reference();
                        s->output_neuron = nullptr;
                    }
                }
            });

    auto it = std::remove_if(this->outgoing_synapses.begin(), this->outgoing_synapses.end(), to_delete_ss);
    this->outgoing_synapses.erase(it, this->outgoing_synapses.end());

    std::for_each(
//            std::execution::seq,
            this->incoming_synapses.begin(),
            this->incoming_synapses.end(),
            [&](synapse *s) {
                if(s->useless) {
                    s->decrement_reference();
                    if(s->input_neuron != nullptr) {
                        s->input_neuron->decrement_reference();
                        s->input_neuron = nullptr;
                    }
                    if(s->output_neuron != nullptr) {
                        s->output_neuron->decrement_reference();
                        s->output_neuron = nullptr;
                    }
                }
            });
    it = std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete_ss);
    this->incoming_synapses.erase(it, this->incoming_synapses.end());

}
//neuron::~neuron() {
//
//    std::cout << "Calling neuron destructor\n";
//    exit(1);
//}
void neuron::fire(int time_step) {
//    Temp hack
    if(this->past_activations.size() > 50){
        this->past_activations.pop();
//        std::cout << "Too many past activations\t" << this->neuron_id << std::endl;
//        exit(1);
    }
    if(this->error_gradient.size() > 50){
        this->error_gradient.pop();
    }
    this->value = temp_value;
    if (this->activation_type && this->value <= 0) {
        this->value = 0;
    } else {
        this->average_activation = this->average_activation * 0.95 + 0.05 * std::abs(this->value);
    }
    temp_value = 0;
    auto activation_val = std::pair<float, int>(this->value, time_step);
    this->past_activations.push(activation_val);
    for (auto it: this->outgoing_synapses)
        it->weight_assignment_past_activations.push(activation_val);
}

void neuron::forward_gradients() {

    if (!this->error_gradient.empty()) {
        for (auto &it : this->incoming_synapses) {

            float message_value;

            message_value = this->error_gradient.front().gradient;

            message grad_temp(message_value, this->error_gradient.front().time_step);
            grad_temp.lambda = this->error_gradient.front().lambda;
            grad_temp.gamma = this->error_gradient.front().gamma;
            grad_temp.error = this->error_gradient.front().error;
            grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
            if (it->pass_gradients)
                it->grad_queue.push(grad_temp);
            it->grad_queue_weight_assignment.push(grad_temp);


        }
        this->error_gradient.pop();
    }
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

        message m(1, time_step);
        m.lambda = lambda;
        m.gamma = gamma;
        m.error = error_grad;

        this->error_gradient.push(m);
        this->past_activations.pop();
        return error * error;
    }
    return 0;
}
//

float neuron::introduce_targets(float target, int time_step, float gamma, float lambda, bool no_grad) {
    //no_grad = false for actions taken
    if (!this->past_activations.empty()) {
        float error = 0;
        if (!no_grad)
            error = target - this->past_activations.front().first;
        float error_grad = error;
//        If activation was zero, we applied relu. Gradients don't flow back
        if (this->past_activations.front().first <= 0 and this->activation_type) {
            std::cout << "Should never get here\n";
            exit(1);
            error_grad = 0;
        }

        message m(int(!no_grad), time_step);
        m.lambda = lambda;
        m.gamma = gamma;
        m.error = error_grad;

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
    std::vector<float> error_vector;
    std::vector<message> messages_q;
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
                    while (!output_synapses_iterator->grad_queue.empty() and !this->past_activations.empty() and
                           this->past_activations.front().second >
                           (output_synapses_iterator->grad_queue.front().time_step -
                            output_synapses_iterator->grad_queue.front().distance_travelled - 1)) {
//                        Activation for this gradient is not stored; this can happen in the beginning of the network initalization, or if new paths are introduced at run time. We just drop this gradient.
//                        if(this->past_activations.front().second > (output_synapses_iterator->grad_queue.front().time_step -
//                                                                   output_synapses_iterator->grad_queue.front().distance_travelled - 1))
//                        {
//                            this->past_activations.pop();
//                        }
//                        else{
                        output_synapses_iterator->grad_queue.pop();
//                        }

                    }
                    if (this->past_activations.empty())
                        return;
                    bool temp_flag = true;
                    if (output_synapses_iterator->grad_queue.empty()) {
//                        "Waiting for gradient from other paths; skipping propagation"
                        temp_flag = false;
                    }

                    if (temp_flag) {
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
                            error_vector.push_back(output_synapses_iterator->grad_queue.front().error);
                            messages_q.push_back(output_synapses_iterator->grad_queue.front());
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
            for (auto &it: this->outgoing_synapses) {
                if (!it->grad_queue.empty() and !wait) {
                    it->grad_queue.pop();
                }
            }


            float err = error_vector[0];
            for (int a = 0; a < error_vector.size(); a++) {
                if (error_vector[a] != err) {
                    std::cout << "Weight = " << this->average_activation << std::endl;
                    std::cout << "Neuron.cpp : Shouldn't happen\n";
                    exit(1);
                }
            }

            message n_message(accumulate_gradient, time_vector[0]);
            n_message.error = error_vector[0];
            n_message.gamma = messages_q[0].gamma;
            n_message.lambda = messages_q[0].lambda;
            auto it = std::max_element(distance_vector.begin(), distance_vector.end());
            n_message.distance_travelled = *it;
            this->past_activations.pop();
            this->error_gradient.push(n_message);

        }
    }
}

int neuron::neuron_id = 0;
normal_random neuron::normal_dist = normal_random(0, 1, 0.1);
