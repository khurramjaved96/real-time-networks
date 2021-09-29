//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/nn/synced_neuron.h"
#include <assert.h>
#include <iostream>
#include <utility>
#include <random>
#include <algorithm>
#include <vector>
#include "../../include/utils.h"

SyncedNeuron::SyncedNeuron(bool is_input, bool is_output) {
  value = 0;
  value_before_firing = 0;
  id = neuron_id_generator;
  useless_neuron = false;
  this->average_activation = 0;
  neuron_id_generator++;
  this->is_output_neuron = is_output;
  is_input_neuron = is_input;
  memory_made = 0;
  neuron_age = 0;
  is_mature = false;
  references = 0;
  neuron_utility = 0;
  drinking_age = 5000;
  mark_useless_prob = 0.999;
  is_bias_unit = false;
}


void SyncedNeuron::set_layer_number(int layer) {
  this->layer_number = layer;
}

int SyncedNeuron::get_layer_number() {
  return this->layer_number;
}

void SyncedNeuron::update_utility() {

  this->neuron_utility = 0;
  for (auto it: this->outgoing_synapses) {
    this->neuron_utility += it->synapse_utility_to_distribute;
  }
  if (this->is_output_neuron)
    this->neuron_utility = 1;

  this->sum_of_utility_traces = 0;
  for (auto it: this->incoming_synapses) {
    if (!it->disable_utility)
      this->sum_of_utility_traces += it->synapse_local_utility_trace;
  }
}

void SyncedNeuron::fire(int time_step) {

  this->update_utility();

//  Forward applies the non-linearity
  this->old_value = this->value;
  this->old_value_without_activation = this->value_without_activation;
  this->value = this->forward(value_before_firing);
  this->value_without_activation = value_before_firing;
  this->shadow_error_prediction = shadow_error_prediction_before_firing;

  value_before_firing = 0;
  shadow_error_prediction_before_firing = 0;

//  Record this activation for gradient calculation purposes
//    auto activation_val = std::pair<float, int>(this->gradient_activation, time_step);


  message_activation activation_val;
  activation_val.gradient_activation = this->backward(this->value);
  activation_val.value_at_activation = this->value;
  activation_val.time = time_step;
  activation_val.error_prediction_value = this->shadow_error_prediction;
//  std::cout << "Outgoing synapses with grad = " << std::endl;
//  std::cout << this->get_no_of_syanpses_with_gradients() << std::endl;
//  if(this->get_no_of_syanpses_with_gradients() > 0) {

  this->past_activations = activation_val;
}

//
void SyncedNeuron::update_value(int time_step) {
  this->neuron_age++;


//  Reset our gradient_activation holder
  this->value_before_firing = 0;
  this->shadow_error_prediction_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    message_activation activation_val;
    activation_val.gradient_activation = it->input_neuron->value;
    activation_val.time = time_step - 1;
    activation_val.error_prediction_value = it->input_neuron->shadow_error_prediction;
    it->weight_assignment_past_activations = activation_val;
//    if(this->id == 9){
//      std::cout << this->value_before_firing << " " << it->weight << " " <<  it->input_neuron->value <<std::endl;
//    }
    if (it->in_shadow_mode) {
      this->shadow_error_prediction_before_firing += it->weight * it->input_neuron->value;
    } else {
      this->value_before_firing += it->weight * it->input_neuron->value;
    }
  }
}

bool to_delete_ss(SyncedSynapse *s) {
  return s->is_useless;
}

/**
 * For each incoming synapse of a neuron, add the gradient from the error in this
 * neuron to its grad_queue for weight assignment. If we do pass gradients backwards,
 * also pass the gradient from the error to grad_queue for use in back propagation.
 */


void SyncedNeuron::forward_gradients() {
//  If this neuron has gradients to pass back
  for (auto &it : this->incoming_synapses) {
    float message_value;

    message_value = this->error_gradient.gradient;

//          We pack our gradient into a new message and pass it back to our incoming synapse.
    message grad_temp(message_value, this->error_gradient.time_step);
    grad_temp.lambda = this->error_gradient.lambda;
    grad_temp.gamma = this->error_gradient.gamma;
    if (it->in_shadow_mode)
      grad_temp.error = this->error_gradient.error_shadow_prediction;
    else
      grad_temp.error = this->error_gradient.error;
    grad_temp.distance_travelled = this->error_gradient.distance_travelled + 1;
    if (it->propagate_gradients)
      it->grad_queue = grad_temp;
    it->grad_queue_weight_assignment = grad_temp;
  }  //  Remove this gradient from our list of things needed to pass back
}

/**
 * NOTE: If you are not VERY familiar with the backprop algorithm, I highly recommend
 * doing some reading before going through this function.
 */


int SyncedNeuron::get_no_of_syanpses_with_gradients() {
  int synapse_with_gradient = 0;
  for (auto it: this->outgoing_synapses) {
    if (it->propagate_gradients)
      synapse_with_gradient++;
  }
  return synapse_with_gradient;
}
void SyncedNeuron::propagate_error() {
  float accumulate_gradient = 0;
  std::vector<int> time_vector;
  std::vector<int> distance_vector;
  std::vector<int> activation_time_required_list;
  std::vector<int> queue_len_vector;
  std::vector<float> error_vector;
  std::vector<message> messages_q;
  int time_check = -99999;

//  No gradient propagation required for prediction nodes

// We need a loop invariant for this function to make sure progress is always made. Things to make sure:
// 1. A queue for a certain outgoing path won't grow large indefinitely
// 2. Adding new connections or removing old connections won't cause deadlocks
// 3. We can never get in a situation in which neither activation nor the gradient is popped. Some number should strictly increase or decrease

// No need to pass gradients if there are no out-going nodes with gradients
  if (this->get_no_of_syanpses_with_gradients() > 0 && !is_input_neuron) {

    for (auto &output_synapses_iterator : this->outgoing_synapses) {

      accumulate_gradient += output_synapses_iterator->weight *
          output_synapses_iterator->grad_queue.gradient *
          this->past_activations.gradient_activation;

      output_synapses_iterator->grad_queue.remove = true;

    }

    message n_message(accumulate_gradient, time_vector[0]);
    n_message.error = error_vector[0];
    n_message.gamma = messages_q[0].gamma;
    n_message.lambda = messages_q[0].lambda;
    auto it = std::max_element(distance_vector.begin(), distance_vector.end());
    n_message.distance_travelled = *it;

    this->error_gradient = n_message;
  }
}

/**
 * Mark synapses and neurons for deletion. Synapses will only get deleted if its age is > 70k.
 * SyncedNeurons will only be deleted if there are no outgoing synapses (and it's not an output neuron of course!)
 */
void SyncedNeuron::mark_useless_weights() {
//  return;
  std::uniform_real_distribution<float> dist(0, 1);
//  std::mt19937 gen;
  float rand_val = dist(SyncedNeuron::gen);
//  std::cout << "Rand value == " << rand_val << std::endl;
  if (this->neuron_age > this->drinking_age * 4) {
    for (auto &it : this->outgoing_synapses) {
//      Only delete weights if they're older than 70k steps
      if (it->output_neuron->neuron_age > it->output_neuron->drinking_age * 4
          && it->synapse_utility < it->utility_to_keep && !it->disable_utility) {
        if (dist(gen) > this->mark_useless_prob)
          it->is_useless = true;
      }
    }
  }

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it a
//  nd its incoming synapses.
  if (this->incoming_synapses.empty() && !this->is_input_neuron) {
    this->useless_neuron = true;
    for (auto it : this->outgoing_synapses)
      it->is_useless = true;
  }


//  if (this->outgoing_synapses.empty() && !this->is_output_neuron && !this->is_input_neuron) {
//    this->useless_neuron = true;
//    for (auto it : this->incoming_synapses)
//      it->is_useless = true;
//  }

  if (this->outgoing_synapses.empty() && !this->is_output_neuron && !this->is_input_neuron) {
    this->useless_neuron = true;
    for (auto it : this->incoming_synapses)
      it->is_useless = true;
  }

}

/**
 * Delete outgoing and incoming synapses that were marked earlier as is_useless.
 */
void SyncedNeuron::prune_useless_weights() {
  std::for_each(
//            std::execution::seq,
      this->outgoing_synapses.begin(),
      this->outgoing_synapses.end(),
      [&](SyncedSynapse*s) {
        if (s->is_useless) {
          s->decrement_reference();
          if (s->input_neuron != nullptr) {
            s->input_neuron->decrement_reference();
            s->input_neuron = nullptr;
          }
          if (s->output_neuron != nullptr) {
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
      [&](SyncedSynapse*s) {
        if (s->is_useless) {
          s->decrement_reference();
          if (s->input_neuron != nullptr) {
            s->input_neuron->decrement_reference();
            s->input_neuron = nullptr;
          }
          if (s->output_neuron != nullptr) {
            s->output_neuron->decrement_reference();
            s->output_neuron = nullptr;
          }
        }
      });
  it = std::remove_if(this->incoming_synapses.begin(), this->incoming_synapses.end(), to_delete_ss);
  this->incoming_synapses.erase(it, this->incoming_synapses.end());
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @return: squared error
 */
float SyncedNeuron::introduce_targets(float target, int time_step) {
//


//      The activation is the output of our NN.
    float error = this->past_activations.value_at_activation - target;
    float error_grad = error;

//      Create our error gradient for this neuron
    message m(error_grad, time_step);
    m.error = 1;
    m.lambda = 0;
    m.gamma = 0;
    this->error_gradient = m;
    return error * error;

  return 0;
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @param gamma: discount factor
 * @param lambda: eligibility trace decay parameter
 * @return: squared error
 */
float SyncedNeuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
//  Introduce a target to a neuron and calculate its error.
//  In this case, target should be our TD target.

//      The activation is the output of our NN.
    float error;
    float error_prediction_error;

    error = this->past_activations.value_at_activation - target;
    error_prediction_error = this->past_activations.error_prediction_value - error;

    float error_grad = error;


//      Create our error gradient for this neuron
    message m(1, time_step);
    m.lambda = lambda;
    m.gamma = gamma;
    m.error = error_grad;
    m.error_shadow_prediction = error_prediction_error;

    this->error_gradient = m;
    return error;

  return 0;
}


float LinearSyncedNeuron::forward(float temp_value) {
//  if (temp_value != 0)
//    this->average_activation = this->average_activation * 0.99 + 0.01 * std::abs(temp_value);
  return temp_value;
}

float LinearSyncedNeuron::backward(float post_activation) {
  return 1;
}

float ReluSyncedNeuron::forward(float temp_value) {

  if (temp_value <= 0)
    return 0;
//  this->average_activation = this->average_activation * 0.9 + 0.1 * std::abs(temp_value);
  return temp_value;
}
//
float ReluSyncedNeuron::backward(float post_activation) {
  if (post_activation > 0)
    return 1;
  else
    return 0;
}

float BiasSyncedNeuron::forward(float temp_value) {
  return 1;
}

float BiasSyncedNeuron::backward(float output_grad) {
  return 0;
}

float LTUSynced::forward(float temp_value) {
  if (temp_value > this->activation_threshold)
    return 1;
  return 0;
}

float LTUSynced::backward(float output_grad) {
  return 0;
}

ReluSyncedNeuron::ReluSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {}

LTUSynced::LTUSynced(bool is_input, bool is_output, float threshold) : SyncedNeuron(is_input, is_output) {
  this->activation_threshold = threshold;
}

BiasSyncedNeuron::BiasSyncedNeuron() : SyncedNeuron(false, false) {
  this->is_bias_unit = true;
  this->average_activation = 1;
}

LinearSyncedNeuron::LinearSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {}

std::mt19937 SyncedNeuron::gen = std::mt19937(0);

int64_t SyncedNeuron::neuron_id_generator = 0;
