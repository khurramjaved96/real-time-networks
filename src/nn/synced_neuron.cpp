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
  neuron_id_generator++;
  this->is_output_neuron = is_output;
  is_input_neuron = is_input;
  neuron_age = 0;
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
  this->neuron_age++;
  this->value = this->forward(value_before_firing);
}

void SyncedNeuron::update_value(int time_step) {

  this->value_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    this->value_before_firing += it->weight * it->input_neuron->value;
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
    grad_temp.error = this->error_gradient.error;

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
  std::vector<float> error_vector;
  std::vector<message> messages_q;

//  No gradient propagation required for prediction nodes

// We need a loop invariant for this function to make sure progress is always made. Things to make sure:
// 1. A queue for a certain outgoing path won't grow large indefinitely
// 2. Adding new connections or removing old connections won't cause deadlocks
// 3. We can never get in a situation in which neither activation nor the gradient is popped. Some number should strictly increase or decrease

// No need to pass gradients if there are no out-going nodes with gradients
  if (this->get_no_of_syanpses_with_gradients() > 0 && !is_input_neuron) {

    for (auto &output_synapses_iterator : this->outgoing_synapses) {
//      std::cout << "Iterating over outoging synapses\n";
//      std::cout << output_synapses_iterator->id << " " << output_synapses_iterator->weight << std::endl;
      accumulate_gradient += output_synapses_iterator->weight *
          output_synapses_iterator->grad_queue.gradient *
          this->backward(this->value);
      error_vector.push_back(output_synapses_iterator->grad_queue.error);
      messages_q.push_back(output_synapses_iterator->grad_queue);
      time_vector.push_back(output_synapses_iterator->grad_queue.time_step);
      output_synapses_iterator->grad_queue.remove = true;

    }

//    std::cout <<
    message n_message(accumulate_gradient, time_vector[0]);
    n_message.error = error_vector[0];
    n_message.gamma = messages_q[0].gamma;
    n_message.lambda = messages_q[0].lambda;
//    auto it = std::max_element(distance_vector.begin(), distance_vector.end());
//    n_message.distance_travelled = *it;

    this->error_gradient = n_message;
  }
}

/**
 * Mark synapses and neurons for deletion. Synapses will only get deleted if its age is > 70k.
 * SyncedNeurons will only be deleted if there are no outgoing synapses (and it's not an output neuron of course!)
 */
void SyncedNeuron::mark_useless_weights() {
//  if (this->is_output_neuron || this->is_input_neuron)
//    return;
  std::uniform_real_distribution<float> dist(0, 1);

  if (this->neuron_age > this->drinking_age) {
    for (auto &it : this->outgoing_synapses) {

      if (it->output_neuron->neuron_age > it->output_neuron->drinking_age
          && it->synapse_utility < it->utility_to_keep && !it->disable_utility) {
        if (dist(gen) > this->mark_useless_prob)
          it->is_useless = true;
      }
    }
  }

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it a
//  nd its incoming synapses.
  if (this->incoming_synapses.empty() && !this->is_input_neuron && !this->is_output_neuron) {
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

  float error = this->value - target;

  message m(this->backward(this->value), time_step);
  m.error = error;
  m.lambda = 0;
  m.gamma = 0;
  this->error_gradient = m;
  return error * error;
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
//float SyncedNeuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
////  Introduce a target to a neuron and calculate its error.
////  In this case, target should be our TD target.
//
////      The activation is the output of our NN.
//  float error;
//
//  error = this->value - target;
//  float error_grad = error;
//
//
////      Create our error gradient for this neuron
//  message m(1, time_step);
//  m.lambda = lambda;
//  m.gamma = gamma;
//  m.error = error_grad;
//
//  this->error_gradient = m;
//  return error;
//}

float LinearSyncedNeuron::forward(float temp_value) {
  return temp_value;
}

float LinearSyncedNeuron::backward(float post_activation) {
  return 1;
}

float ReluSyncedNeuron::forward(float temp_value) {

  if (temp_value <= 0)
    return 0;

  return temp_value;
}
//
float ReluSyncedNeuron::backward(float post_activation) {
  if (post_activation > 0)
    return 1;
  else
    return 0;
}

float SigmoidSyncedNeuron::forward(float temp_value) {

  float post_activation = sigmoid(temp_value);
  return post_activation;
}

float SigmoidSyncedNeuron::backward(float post_activation) {
  return post_activation * (1 - post_activation);
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

SigmoidSyncedNeuron::SigmoidSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {}

LTUSynced::LTUSynced(bool is_input, bool is_output, float threshold) : SyncedNeuron(is_input, is_output) {
  this->activation_threshold = threshold;
}

BiasSyncedNeuron::BiasSyncedNeuron() : SyncedNeuron(false, false) {
  this->is_bias_unit = true;
}

LinearSyncedNeuron::LinearSyncedNeuron(bool is_input, bool is_output) : SyncedNeuron(is_input, is_output) {}

std::mt19937 SyncedNeuron::gen = std::mt19937(0);

int64_t SyncedNeuron::neuron_id_generator = 0;
