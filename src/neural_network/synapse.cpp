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
    references = 0;
    input_neuron = input;
    input->increment_reference();
    input_neuron->sucesses++;
    output_neuron = output;
    output->increment_reference();
    credit = 0;
    mark_delete = false;
    enable_logging = false;
    useless = false;
    age = 0;
    credit_activation_idbd = 0;
    weight = w;
    this->step_size = step_size;
    this->increment_reference();
    input_neuron->outgoing_synapses.push_back(this);
    this->increment_reference();
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

//synapse::~synapse(){
//    std::cout << "Destroying synapses\n";
//    exit(1);
//}

void synapse::assign_credit() {
    if(this->grad_queue.size()>50){
        this->grad_queue.pop();
//        std::cout << "Too long queue synapss\n";
//        exit(1);
    }
    if(this->grad_queue_weight_assignment.size() > 50){
        this->grad_queue_weight_assignment.pop();
    }
    if(this->weight_assignment_past_activations.size() > 50){
        this->weight_assignment_past_activations.pop();
    }


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
            this->trace = this->trace * this->grad_queue_weight_assignment.front().gamma *
                          this->grad_queue_weight_assignment.front().lambda +
                          this->weight_assignment_past_activations.front().first*this->grad_queue_weight_assignment.front().gradient;

            this->credit =  this->trace * this->grad_queue_weight_assignment.front().error;


            this->credit_activation_idbd = this->weight_assignment_past_activations.front().first;
            this->grad_queue_weight_assignment.pop();
            this->weight_assignment_past_activations.pop();

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
//    this->beta_step_size = enable_logging(step_size);
//    this->h_step_size = 0;
//    this->step_size = exp(this->beta_step_size);
}
void synapse::update_weight()
{

    if(this->idbd)
    {
        this->beta_step_size += (1e-4)*this->credit*this->h_step_size;
        this->step_size = exp(this->beta_step_size);
        this->weight += (this->step_size * this->credit);
        this->h_step_size = this->h_step_size *(1 - this->step_size*this->credit_activation_idbd*this->trace) + this->step_size*this->credit;

    }
    else
    {

        this->weight += (this->step_size * this->credit);
    }

}

void synapse::read_gradient() {
    auto grad = this->output_neuron->error_gradient.front();

}


no_grad_synapse::no_grad_synapse(neuron *input, neuron *output) {
    this->input_neurons = input;
    this->output_neurons = output;

}

void no_grad_synapse::copy_activation(int time_step) {
    this->output_neurons->temp_value = this->input_neurons->temp_value;

}
