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

long long int synapse::synapse_id = 0;
synapse::synapse(neuron *input, neuron *output, float w, float step_size) {
    input_neuron = input;
    input_neuron->sucesses++;
    output_neuron = output;
    credit = 0;
    mark_delete = false;
    log = false;
    useless = false;
    age = 0;
    credit_activation_idbd = 0;
    weight = w;
    this->step_size = step_size;
    input_neuron->outgoing_synapses.push_back(this);
    output_neuron->incoming_synapses.push_back(this);
    print_status = false;
    this->idbd = false;
    this->b1 = 0;
    this->id = synapse_id;
    synapse_id++;
    this->b2 = 100;
    trace = 0;
    pass_gradients = true;
}

void synapse::assign_credit() {
    int activation_time_required = this->grad_queue_weight_assignment.front().time_step -
                                   this->grad_queue_weight_assignment.front().distance_travelled - 1;

    while (!this->grad_queue_weight_assignment.empty() and !this->weight_assignment_past_activations.empty() and this->weight_assignment_past_activations.front().second >
                                                                                                 (this->grad_queue_weight_assignment.front().time_step -
                                                                                                  this->grad_queue_weight_assignment.front().distance_travelled - 1)) {

        this->grad_queue_weight_assignment.pop();
    }
    if(!this->grad_queue_weight_assignment.empty() and this->weight_assignment_past_activations.front().second !=  (this->grad_queue_weight_assignment.front().time_step -
                                                                    this->grad_queue_weight_assignment.front().distance_travelled - 1))
    {
        std::cout << "Synapses.cpp : Shouldn't happen\n";
        exit(1);
    }


    if (this->grad_queue_weight_assignment.size() > 0) {
        if (this->output_neuron->is_output_neuron or true) {
            this->trace = this->trace * this->grad_queue_weight_assignment.front().gamma *
                          this->grad_queue_weight_assignment.front().lambda +
                          this->weight_assignment_past_activations.front().first*this->grad_queue_weight_assignment.front().gradient;
//            if (this->log)
//            {
//                std::cout << "Lambda\tGamma\tError\n";
//                std::cout << this->grad_queue_weight_assignment.front().lambda << "\t" << this->grad_queue_weight_assignment.front().gamma << std::endl;
//                std::cout << "Trace\tGrad\tError\n";
//                std::cout << this->trace << "\t" << this->grad_queue_weight_assignment.front().gradient << "\t"
//                          << this->grad_queue_weight_assignment.front().error << std::endl;
//            }
            this->credit =  this->trace * this->grad_queue_weight_assignment.front().error;

            this->credit_activation_idbd = this->weight_assignment_past_activations.front().first;
            this->grad_queue_weight_assignment.pop();
            this->weight_assignment_past_activations.pop();
        } else {

            this->credit = this->grad_queue_weight_assignment.front().gradient *
                         this->weight_assignment_past_activations.front().first * this->grad_queue_weight_assignment.front().error;
            this->credit_activation_idbd = this->weight_assignment_past_activations.front().first;
            this->grad_queue_weight_assignment.pop();
            this->weight_assignment_past_activations.pop();
        }
    } else {
        this->credit = 0;
    }
}

void synapse::reset_trace() {
    this->trace = 0;
}

void synapse::block_gradients() {
    pass_gradients = false;
}

void synapse::zero_gradient() {
    this->credit = 0;
}

void synapse::turn_on_idbd() {
    this->step_size = this->step_size*10;
    return;
//    std::cout << "STEP SIZE = " << this->step_size << std::endl;
//    exit(1);
//    this->idbd = true;
//    this->beta_step_size = log(this->step_size);
////    this->beta_step_size = log(step_size);
//    this->h_step_size = 0;
//    this->step_size = exp(this->beta_step_size);
}
void synapse::update_weight()
{

    if(this->idbd)
    {
        this->beta_step_size += (1e-4)*this->credit*this->h_step_size/(sqrt(this->b2) + 1e-8);
        this->b2 = this->b2*0.99 + this->credit*this->h_step_size*this->credit*this->h_step_size*0.01;
        this->step_size = exp(this->beta_step_size);
//        if(this->credit > 0 and this->step_size > 1e-2)
//            std::cout << "Step_size = " << this->step_size << " " << this->h_step_size << " " << this->credit*this->h_step_size <<  std::endl;
        this->weight += (this->step_size * this->credit);
        this->h_step_size = this->h_step_size *(1 - this->step_size*this->credit_activation_idbd*this->credit_activation_idbd) + this->step_size*this->credit;

    }
    else
    {
        if(this->log)
        std::cout << this->id << " " <<  this->weight <<  " " << this->credit << std::endl;
        this->weight += (this->step_size * this->credit);
    }
//    if(this->credit > 0)
//    {
//        this->weight = this->weight*0.9999;
//    }
//    this->weight = this->weight * 0.9999;
//    if(this->weight > 1){
//        this->weight = 1;
//    }
//    else if (this->weight < -1)
//    {
//        this->weight = -1;
//    }
////            this->credit = 0;
//    if(this->credit != 0) {
////        this->b1 = this->b1 * 0.9 + this->credit * 0.1;
//        this->b2 = this->b2 * 0.99 + (this->credit * this->credit) * 0.01;
//        this->weight += (this->step_size * this->credit) / (sqrt(this->b2) + 1e-8);
//    }
}

void synapse::read_gradient() {
    auto grad = this->output_neuron->error_gradient.front();

}


no_grad_synapse::no_grad_synapse(neuron *input, neuron *output) {
    this->input_neurons = input;
    this->output_neurons = output;

}

void no_grad_synapse::copy_activation(int time_step) {
//    if(this->input_neurons->value > 0)
//        std::cout << "Copying val " << this->input_neurons->value << " to " << this->output_neurons->temp_value << std::endl;
//    std::cout << "No grad synapse value  =" <<

    this->output_neurons->temp_value = this->input_neurons->temp_value;
//    this->output_neurons->past_activations.push(std::pair<float, int>(this->output_neurons->value, time_step));
}
