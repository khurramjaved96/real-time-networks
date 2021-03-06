//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/nn/synapse.h"
#include <math.h>
#include <vector>
#include <iostream>
#include "../../include/nn/neuron.h"
#include "../../include/nn/utils.h"


int64_t synapse::synapse_id_generator = 0;

synapse::synapse(Neuron *input, Neuron *output, float w, float step_size) {
  references = 0;
  TH = 0;

  this->is_recurrent_connection = false;
  input_neuron = input;
  input->increment_reference();
  input_neuron->sucesses++;
  output_neuron = output;
  output->increment_reference();
  credit = 0;
  this->print_status = false;
  in_shadow_mode = false;
  is_useless = false;
  age = 0;
  weight = w;
  this->step_size = step_size;
  this->increment_reference();
  input_neuron->outgoing_synapses.push_back(this);
  this->increment_reference();
  output_neuron->incoming_synapses.push_back(this);
  this->idbd = false;
  this->id = synapse_id_generator;
  synapse_id_generator++;
  this->l2_norm_meta_gradient = 100;
  trace = 0;
  propagate_gradients = true;
  synapse_utility = 0;
  meta_step_size = 1e-4;
  if (input->is_input_neuron) {
    propagate_gradients = false;
  }
  this->synapse_local_utility_trace = 0;
  utility_to_keep = 0.0001;
  disable_utility = false;
  synapse_local_utility_trace_decay = 0.99999;
  this->synapse_local_utility_trace = 0;
  this->synapse_utility_to_distribute = 0;
  this->n_feature_activity_contributions = 0;
}
//

void synapse::set_utility_to_keep(float util) {
  this->utility_to_keep = util;
}

float synapse::get_utility_to_keep() {
  return this->utility_to_keep;
}

void synapse::set_connected_to_recurrence(bool val) {
  this->is_recurrent_connection = val;
}

void synapse::set_shadow_weight(bool val) {
  this->in_shadow_mode = val;
}

void synapse::reset_trace() {
  this->trace = 0;
}

void synapse::set_meta_step_size(float val) {
  this->meta_step_size = val;
}

void synapse::update_utility() {
  float diff = this->output_neuron->value - this->output_neuron->forward(
      this->output_neuron->value_without_activation - this->input_neuron->old_value * this->weight);
//  0.999 is a hyper-parameter.
  if(!this->in_shadow_mode && !this->disable_utility) {
    this->synapse_local_utility_trace =
      this->synapse_local_utility_trace_decay * this->synapse_local_utility_trace
      + (1 - this->synapse_local_utility_trace_decay) * std::abs(diff);
    //if (this->id == 2)
    //  std::cout << ">> " << synapse_local_utility_trace << "*" << this->output_neuron->neuron_utility << "/" << this->output_neuron->sum_of_utility_traces + 1e-10<< std::endl;
    this->synapse_utility =
        (synapse_local_utility_trace * this->output_neuron->neuron_utility)
            / (this->output_neuron->sum_of_utility_traces + 1e-10);
    if (this->synapse_utility > this->utility_to_keep) {
      this->synapse_utility_to_distribute = this->synapse_utility - this->utility_to_keep;
      this->synapse_utility = this->utility_to_keep;
    } else {
      this->synapse_utility_to_distribute = 0;
    }
  }
  else{
    this->synapse_utility = 0;
    this->synapse_utility_to_distribute = 0;
    this->synapse_local_utility_trace = 0;
  }
}

void synapse::memory_leak_patch(){
  if (this->grad_queue.size() > 50) {
    this->grad_queue.pop();
  }
  if (this->grad_queue_weight_assignment.size() > 50) {
    this->grad_queue_weight_assignment.pop();
  }
  if (this->weight_assignment_past_activations.size() > 50) {
    this->weight_assignment_past_activations.pop();
  }
}

/**
 * Calculate and set credit based on gradients in the current synapse.
 */
void synapse::assign_credit() {

//  Another temp hack


//  We go through each gradient that we've put into our synapse
//  and see if this gradient's activation time corresponds to the correct past activation
  while (!this->grad_queue_weight_assignment.empty() && !this->weight_assignment_past_activations.empty() &&
      this->weight_assignment_past_activations.front().time >
          (this->grad_queue_weight_assignment.front().time_step -
              this->grad_queue_weight_assignment.front().distance_travelled - 1)) {
//      If it doesn't then remove it
    this->grad_queue_weight_assignment.pop();
  }

//  If this condition is met, your gradient flew past its relevant activation - this isn't supposed to happen!
  if (!this->grad_queue_weight_assignment.empty() && this->weight_assignment_past_activations.front().time !=
      (this->grad_queue_weight_assignment.front().time_step -
          this->grad_queue_weight_assignment.front().distance_travelled - 1)) {
    if (this->is_recurrent_connection) {
      std::cout << "Is recurrent connection\n";
    }
    std::cout << "Synapses.cpp : Shouldn't happen\n";
    exit(1);
  }

//  If we still have gradients left for credit assignment
  if (!this->grad_queue_weight_assignment.empty()) {
//      We have a match! Here we calculate our update rule. We first update our eligibility trace]

    this->trace += this->weight_assignment_past_activations.front().gradient_activation *
            this->grad_queue_weight_assignment.front().gradient;
//    std::cout << "Gamma\t" << this->grad_queue_weight_assignment.front().gamma << " Lambda \t" << this->grad_queue_weight_assignment.front().lambda << std::endl;
    this->tidbd_old_activation = this->weight_assignment_past_activations.front().gradient_activation;
    this->tidbd_old_error = this->grad_queue_weight_assignment.front().error;

//      As per the trace update rule, our actual credit being assigned is our trace x our TD error.
    this->credit = this->trace * this->grad_queue_weight_assignment.front().error;

    this->trace = this->trace * this->grad_queue_weight_assignment.front().gamma *
        this->grad_queue_weight_assignment.front().lambda;

//      Remove both grad and past activations used
    this->grad_queue_weight_assignment.pop();
    this->weight_assignment_past_activations.pop();

  } else {
    this->credit = 0;
  }
}

void synapse::block_gradients() {
  propagate_gradients = false;
}

bool synapse::get_recurrent_status() {
  return is_recurrent_connection;
}

void synapse::turn_on_idbd() {
  this->idbd = true;
  this->log_step_size_tidbd = log(this->step_size);
  this->h_tidbd = 0;
  this->step_size = exp(this->log_step_size_tidbd);
}
//
void synapse::turn_off_idbd() {
  this->idbd = false;
}

void synapse::update_weight() {
//
  if (this->idbd) {
    float meta_grad = this->tidbd_old_error * this->trace * this->h_tidbd;
    this->l2_norm_meta_gradient = this->l2_norm_meta_gradient * 0.99 + (1 - 0.99) * (meta_grad * meta_grad);
    if (age > 1000) {
      this->log_step_size_tidbd += this->meta_step_size * meta_grad / (sqrt(this->l2_norm_meta_gradient) + 1e-8);
      this->log_step_size_tidbd = max(this->log_step_size_tidbd, -15);
      this->log_step_size_tidbd = min(this->log_step_size_tidbd, -3);
      this->step_size = exp(this->log_step_size_tidbd);
      this->weight -= (this->step_size * this->credit);
      if ((1 - this->step_size * this->tidbd_old_activation * this->trace) > 0) {
        this->h_tidbd =
            this->h_tidbd * (1 - this->step_size * this->tidbd_old_activation * this->trace) +
                this->step_size * this->trace * this->tidbd_old_error;
//        std::cout << "Decay rate " << (1 - this->step_size * this->tidbd_old_activation * this->trace) << std::endl;
      } else {
        this->h_tidbd = this->step_size * this->trace * this->tidbd_old_error;
      }
    }

  } else {
    this->weight -= (this->step_size * this->credit);
  }
  if (this->weight > 5)
    this->weight = 5;
  if (this->weight < -5)
    this->weight = -5;

  if (this->is_recurrent_connection) {
    if (this->weight > 0.9) {
      this->weight = 0.9;
    }
    if (this->weight < 0) {
      this->weight = 0;
    }
  }
}

