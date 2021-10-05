//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/nn/neuron.h"
#include <assert.h>
#include <iostream>
#include <utility>
#include <random>
#include <algorithm>
#include <vector>
#include "../../include/utils.h"



Neuron::Neuron(bool is_input, bool is_output) {
  value = 0;
  old_value = 0;
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

/**
 * Fire a neuron. Use the update_value calculated gradient_activation to set this->gradient_activation to
 * the activation by applying an activation function (in this case ReLU) to the calculated gradient_activation.
 * @param time_step: time step that this neuron fires. Used for recording our activation gradient_activation firing time.
 */



void Neuron::update_utility() {

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

void Neuron::memory_leak_patch() {
  if (this->past_activations.size() > 200) {
    this->past_activations.pop();
  }
  if (this->error_gradient.size() > 200) {
    this->error_gradient.pop();
  }
}

void Neuron::fire(int time_step) {

  this->update_utility();
  this->memory_leak_patch();

//  Forward applies the non-linearity
  if (!this->is_input_neuron){
    // this is already updated in set_input_values
    this->old_old_value = this->old_value;
    this->old_value = this->value;
    this->old_value_without_activation = this->value_without_activation;
    // value always end up being set to zeros for input_neurons, dont want that
    this->value = this->forward(value_before_firing);
  }
  if(this->value > this->average_activation){
    this->average_activation = this->value;
  }
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
  if (!this->is_input_neuron)
    this->past_activations.push(activation_val);
//  }

}

/**
 * For this neuron, calculate the outgoing gradient_activation (pre activation function) for this time step and set it to
 * value_before_firing.
 * Additionally, when the neuron reaches maturity (age >= 20k), scale the
 * incoming weights so the current node's incoming activation is on average 1, and scale
 * the outgoing weights so that the outgoing activation stays the same.
 */

float Neuron::backward_credit(float activation_value, synapse *it) {
  return activation_value;
}

//float RecurrentNeuron::backward_credit(float activation_value, synapse *it) {
//
//  RecurrentNeuron *p = static_cast<RecurrentNeuron*>(it->output_neuron);
//  assert(it->output_neuron->is_output_neuron == false);
//  it->TH = activation_value + p->recurrent_synapse->weight * it->TH;
//  assert(!it->get_recurrent_status() || it->id == p->recurrent_synapse->id);
//  return it->TH;
//}

//
void Neuron::update_value(int time_step) {
  this->neuron_age++;


//  Reset our gradient_activation holder
  this->value_before_firing = 0;
  this->shadow_error_prediction_before_firing = 0;

  //this->normalize_neuron();

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    message_activation activation_val;
    activation_val.gradient_activation = it->input_neuron->value;
    activation_val.time = time_step - 1;
    activation_val.error_prediction_value = it->input_neuron->shadow_error_prediction;
    it->weight_assignment_past_activations.push(activation_val);
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

void Neuron::normalize_neuron() {

  if (this->neuron_age == this->drinking_age * 4 && !this->is_output_neuron) {
    this->is_mature = true;
    for (auto it : this->incoming_synapses) {
      if (!it->get_recurrent_status()) {
        it->step_size = 0;
        it->turn_off_idbd();
      }
    }

  }
  if(this->neuron_age % this->drinking_age == 0 && !this->is_input_neuron && !this->is_output_neuron){

    if(this->average_activation < 0){
      std::cout << "neuron:cpp: Negative max value; shouldn't happen\n";
      exit(1);
    }
//  }
//
//  if (this->neuron_age == this->drinking_age && !this->is_input_neuron && !this->is_output_neuron) {

    float scale = 1 / this->average_activation;
    if(scale > 2 or this->average_activation == 0){
      scale = 2;
    }
    else if(scale < 0.2){
      scale = 0.2;
    }
    for (auto it : this->incoming_synapses) {
      if (!it->get_recurrent_status()) {
        it->weight = it->weight * scale;
      }
    }


    for (auto out_g : this->outgoing_synapses) {
      if (!out_g->get_recurrent_status()) {
        out_g->set_shadow_weight(false);
        out_g->weight /= scale;
        if(out_g->step_size == 0){
          out_g ->step_size = 1e-5;
        }
        out_g->step_size /= scale;
        out_g->set_meta_step_size(3e-3);
        out_g->turn_on_idbd();
      }
    }
    this->average_activation = -10;
  }
}

bool to_delete_ss(synapse *s) {
  return s->is_useless;
}

/**
 * For each incoming synapse of a neuron, add the gradient from the error in this
 * neuron to its grad_queue for weight assignment. If we do pass gradients backwards,
 * also pass the gradient from the error to grad_queue for use in back propagation.
 */


void Neuron::forward_gradients() {
//  If this neuron has gradients to pass back
  if (!this->error_gradient.empty()) {
//      We do so to all incoming synapses
    for (auto &it : this->incoming_synapses) {
      float message_value;

      message_value = this->error_gradient.front().gradient;

//          We pack our gradient into a new message and pass it back to our incoming synapse.
      message grad_temp(message_value, this->error_gradient.front().time_step);
      grad_temp.lambda = this->error_gradient.front().lambda;
      grad_temp.gamma = this->error_gradient.front().gamma;
      if (it->in_shadow_mode)
        grad_temp.error = this->error_gradient.front().error_shadow_prediction;
      else
        grad_temp.error = this->error_gradient.front().error;
      grad_temp.distance_travelled = this->error_gradient.front().distance_travelled + 1;
      if (it->propagate_gradients)
        it->grad_queue.push(grad_temp);
      it->grad_queue_weight_assignment.push(grad_temp);
    }  //  Remove this gradient from our list of things needed to pass back
    this->error_gradient.pop();
  }
}

/**
 * NOTE: If you are not VERY familiar with the backprop algorithm, I highly recommend
 * doing some reading before going through this function.
 */


int Neuron::get_no_of_syanpses_with_gradients() {
  int synapse_with_gradient = 0;
  for (auto it: this->outgoing_synapses) {
    if (it->propagate_gradients)
      synapse_with_gradient++;
  }
  return synapse_with_gradient;
}
void Neuron::propagate_error() {
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
//    Variables shared across all out-going synapses
    bool flag = false;
    bool wait = false;
    bool propagation = true;
    int total_pops = 0;
    bool all_empty = true;

    for (auto &output_synapses_iterator : this->outgoing_synapses) {
      // Iterate over all outgoing synapses. We want to make sure
//          Skip this if there are no gradients to propagate for this synapse or if we have decided to no propagate gradients for other reasons
      if (!output_synapses_iterator->grad_queue.empty()) {
        all_empty = true;
//              This diff in time_step and distance_travelled is essentially "how long until I activate this gradient"
//              Currently, b/c of grad_temp.distance_travelled = error_gradient.front().distance_travelled + 1
//              this means this will always be this->past_activations.front().second - 2.

//              So now we need to match the right past activation with the activation time required.
//              Since we always truncate gradients after 1 step, this corresponds to having a past activation time
//              the same as the time step the gradient was calculated - 2. grad distance_travelled is
//              always 1 in this case.

//              Remove all past activations that are older than the activation time required of the earliest gradient

//        If we have gradients for which corresponding activations have already been used, pop all gradients until the right activation and then postpone propagation for other nodes
        while (!output_synapses_iterator->grad_queue.empty() && !this->past_activations.empty() &&
            this->past_activations.front().time > output_synapses_iterator->grad_queue.front().time_step -
                output_synapses_iterator->grad_queue.front().distance_travelled - 1) {
          output_synapses_iterator->grad_queue.pop();
          total_pops++;
        }

//              This means all the gradients left past here need to be passed back.

//              We check to see if we have any past activations
        if (this->past_activations.empty())
          return;

//              If we have the situation where an outgoing synapse "skips" neurons
//              This synapse's grad calculation needs to wait until the other chain of neurons
//              is done propagating backwards.
//              grad_queue will be empty in the case that you have a few backprop steps before
//              your corresponding gradient arrives.

        if (output_synapses_iterator->grad_queue.empty()) {
//                  Waiting for gradient from other paths; skipping propagation
//          std::cout << "Wating for other grad from other paths\n";
//        If grad queue was not empty in the beginning, but then was made empty,
//        that means we might have to delay gradient propagation because a new path,
//        further away from the output has been introduced dynamically
          propagation = false;
        }

        if (propagation) {
//                  Here we have gradients to process
          int activation_time_required = output_synapses_iterator->grad_queue.front().time_step -
              output_synapses_iterator->grad_queue.front().distance_travelled - 1;
          activation_time_required_list.push_back(activation_time_required);

          if (this->past_activations.front().time == activation_time_required) {
            time_vector.push_back(output_synapses_iterator->grad_queue.front().time_step);
            distance_vector.push_back(output_synapses_iterator->grad_queue.front().distance_travelled);
            queue_len_vector.push_back(output_synapses_iterator->grad_queue.size());
            error_vector.push_back(output_synapses_iterator->grad_queue.front().error);
            messages_q.push_back(output_synapses_iterator->grad_queue.front());
//
//                      Here we accumulate all our grads wrt the forward node activation
//                      according to the backprop algorithm.
//                      Only accumulate gradient if activation was non-zero.

            accumulate_gradient += output_synapses_iterator->weight *
                output_synapses_iterator->grad_queue.front().gradient *
                this->past_activations.front().gradient_activation;



//                      Check that all activation_time_required are the same
            if (time_check == -99999) {
              time_check = activation_time_required;
            } else {
              if (time_check != activation_time_required) {
                std::cout << "Mismatch between gradient times. Exiting\n";
                exit(1);
                flag = true;
              }
            }
            output_synapses_iterator->grad_queue.front().remove = true;
          }
        }
      }
    }

    if ((total_pops == 0 || all_empty) && !this->past_activations.empty()) {
//      To satisfy the loop invariant condition in a corner case
//      this->past_activations.pop();
    }
    if (flag || time_vector.empty() || !propagation)
      return;

//      Remove all the grads we just processed
//    At-least one out-going connection that is popped must be empty
    bool at_least_one_empty = false;
    for (auto &it : this->outgoing_synapses) {
      if (!it->grad_queue.empty() && !wait && it->grad_queue.front().remove) {
        it->grad_queue.pop();
        if (it->grad_queue.empty())
          at_least_one_empty = true;
      }
    }
    if (!at_least_one_empty) {
      std::cout << "Unnecessary delay. Can mess with gradient alignment. Quitting\n";
      exit(1);
    }



//      check all errors are the same (from the same target)
    float err = error_vector[0];
    for (int a = 0; a < error_vector.size(); a++) {
      if (error_vector[a] != err) {
        std::cout << "Average activation = " << this->average_activation << std::endl;
        std::cout << "Neuron.cpp : Shouldn't happen\n";
        std::cout << error_vector[a] << " " << err << std::endl;
        exit(1);
      }
    }

//      Now we make a message to pass our grad of our loss w.r.t. this activation to this neuron
    message n_message(accumulate_gradient, time_vector[0]);
    n_message.error = error_vector[0];
    n_message.gamma = messages_q[0].gamma;
    n_message.lambda = messages_q[0].lambda;
    auto it = std::max_element(distance_vector.begin(), distance_vector.end());
    n_message.distance_travelled = *it;

//      Remove the activation we just processed
    this->past_activations.pop();
    this->error_gradient.push(n_message);
  }
}

/**
 * Mark synapses and neurons for deletion. Synapses will only get deleted if its age is > 70k.
 * Neurons will only be deleted if there are no outgoing synapses (and it's not an output neuron of course!)
 */
void Neuron::mark_useless_weights() {
//  return;
  std::uniform_real_distribution<float> dist(0, 1);
//  std::mt19937 gen;
  float rand_val = dist(Neuron::gen);
//  std::cout << "Rand value == " << rand_val << std::endl;
  if(this->neuron_age > this->drinking_age * 4) {
    for (auto &it : this->outgoing_synapses) {
//      Only delete weights if they're older than 70k steps
      if (it->output_neuron->neuron_age > it->output_neuron->drinking_age * 4 && it->synapse_utility < it->utility_to_keep && !it->disable_utility) {
        if (dist(gen) > this->mark_useless_prob)
          it->is_useless = true;
      }
    }
  }

//  if this current neuron has no outgoing synapses and is not an output or input neuron,
//  delete it a
//  nd its incoming synapses.
  if(this->incoming_synapses.empty() && !this->is_input_neuron && !this->is_bias_unit){
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
void Neuron::prune_useless_weights() {
  std::for_each(
//            std::execution::seq,
      this->outgoing_synapses.begin(),
      this->outgoing_synapses.end(),
      [&](synapse *s) {
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
      [&](synapse *s) {
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
float Neuron::introduce_targets(float target, int time_step) {
//

  if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
    float error = this->past_activations.front().value_at_activation - target;
    float error_grad = error;

//      Create our error gradient for this neuron
    message m(error_grad, time_step);
    m.error = 1;
    m.lambda = 0;
    m.gamma = 0;
    this->error_gradient.push(m);
    this->past_activations.pop();
    return error * error;
  }
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
float Neuron::introduce_targets(float target, int time_step, float gamma, float lambda) {
//  Introduce a target to a neuron and calculate its error.
//  In this case, target should be our TD target.

  if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
    float error;
    float error_prediction_error;

    error = this->past_activations.front().value_at_activation - target;
    error_prediction_error = this->past_activations.front().error_prediction_value - error;

    float error_grad = error;


//      Create our error gradient for this neuron
    message m(1, time_step);
    m.lambda = lambda;
    m.gamma = gamma;
    m.error = error_grad;
    m.error_shadow_prediction = error_prediction_error;

    this->error_gradient.push(m);
    this->past_activations.pop();
    return error;
  }
  return 0;
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @param gamma: discount factor
 * @param lambda: eligibility trace decay parameter
 * @param no_grad: whether grad is computed
 * @return: squared error
 */
float Neuron::introduce_targets(float target, int time_step, float gamma, float lambda, bool no_grad) {
//  Introduce a target to a neuron and calculate its error.
//  In this case, target should be our TD target.

  if (!this->past_activations.empty()) {
//      The activation is the output of our NN.
    float error = 0;
    float error_prediction_error;
    if (!no_grad)
      error = this->past_activations.front().value_at_activation - target;
    error_prediction_error = this->past_activations.front().error_prediction_value - error;

    float error_grad = error;


//      Create our error gradient for this neuron
    message m(static_cast<int>(!no_grad), time_step);
    m.lambda = lambda;
    m.gamma = gamma;
    m.error = error_grad;
    m.error_shadow_prediction = error_prediction_error;

    this->error_gradient.push(m);
    this->past_activations.pop();
    return error;
  }
  return 0;
}

float SigmoidNeuron::forward(float temp_value) {

  float post_activation = sigmoid(temp_value);
//  this->average_activation = this->average_activation * 0.99 + 0.01 * std::abs(post_activation);
  return post_activation;
}

float SigmoidNeuron::backward(float post_activation) {
  return post_activation * (1 - post_activation);
}

float Tanh::forward(float temp_value) {
  float post_activation = tanh(temp_value);
//  this->average_activation = this->average_activation * 0.99 + 0.01 * std::abs(post_activation);
  return post_activation;
}

float Tanh::backward(float output_grad) {
  return 1 - (output_grad*output_grad);
}


float LinearNeuron::forward(float temp_value) {
//  if (temp_value != 0)
//    this->average_activation = this->average_activation * 0.99 + 0.01 * std::abs(temp_value);
  return temp_value;
}

float LinearNeuron::backward(float post_activation) {
  return 1;
}

float ReluNeuron::forward(float temp_value) {

  if (temp_value <= 0)
    return 0;
//  this->average_activation = this->average_activation * 0.9 + 0.1 * std::abs(temp_value);
  return temp_value;
}
//
float ReluNeuron::backward(float post_activation) {
  if (post_activation > 0)
    return 1;
  else
    return 0;
}
//
float ReluThresholdNeuron::forward(float temp_value) {
  if (temp_value <= this->threshold)
    return 0;
  this->average_activation = 1;
  return 1;
}

float ReluThresholdNeuron::backward(float post_activation) {
  if (post_activation > threshold)
    return 1;
  else
    return 0;
}

float LeakyRelu::forward(float temp_value) {
  if (temp_value < 0)
    return this->negative_slope * temp_value;
  return temp_value;
}

float LeakyRelu::backward(float post_activation) {
  if (post_activation >= 0)
    return 1;
  else
    return this->negative_slope;
}

float BiasNeuron::forward(float temp_value) {
  return 1;
}


float BiasNeuron::backward(float output_grad) {
  return 0;
}

float LTU::forward(float temp_value) {
  if(temp_value > this->activation_threshold)
    return 1;
  return 0;
}

float LTU::backward(float output_grad) {
  return 0;
}

ReluNeuron::ReluNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

LTU::LTU(bool is_input, bool is_output, float threshold): Neuron(is_input, is_output) {
  this->activation_threshold = threshold;
}


ReluThresholdNeuron::ReluThresholdNeuron(float threshold) : Neuron(false, false) {
  this->threshold = threshold;
}

BiasNeuron::BiasNeuron() : Neuron(false, false) {
  this->is_bias_unit = true;
  this->average_activation = 1;
}

SigmoidNeuron::SigmoidNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

Tanh::Tanh(bool is_input, bool is_output) : Neuron(is_input, is_output){}

LeakyRelu::LeakyRelu(bool is_input, bool is_output, float negative_slope) : Neuron(is_input, is_output) {
  this->negative_slope = negative_slope;
}

LinearNeuron::LinearNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}
//
//RecurrentNeuron::RecurrentNeuron(bool is_input, bool is_output, synapse *recurrent_synapse) : Neuron(is_input,
//                                                                                                     is_output) {
//  this->is_recurrent_neuron = true;
//  this->recurrent_synapse = recurrent_synapse;
//  this->recurrent_synapse->input_neuron = this;
//  this->recurrent_synapse->output_neuron = this;
//  this->recurrent_synapse->set_connected_to_recurrence(true);
//}
//
//RecurrentReluNeuron::RecurrentReluNeuron(bool is_input, bool is_output, synapse *recurrent_synapse) : RecurrentNeuron(
//    is_input,
//    is_output,
//    recurrent_synapse) {}
//
//RecurrentSigmoidNeuron::RecurrentSigmoidNeuron(bool is_input, bool is_output, synapse *recurrent_synapse)
//    : RecurrentNeuron(is_input, is_output, recurrent_synapse) {}
//

BoundedNeuron::BoundedNeuron(bool is_input, bool is_output, float bound_replacement_prob, float bound_max_range) : Neuron(is_input, is_output) {
  this->num_times_reassigned = 0;
  this->mark_useless_prob = bound_replacement_prob;
  this->bound_max_range = bound_max_range;
}

void BoundedNeuron::update_activation_bounds(synapse * incoming_synapse) {
  // find a random center point and the find random lower and upper bounds that are close to the center point.
  if (incoming_synapse->weight != 1){
    std::cout << "neuron.cpp: BoundedNeuron bounds are based on incoming neuron value ranges" << std::endl;
    exit(1);
  }

  std::pair<float,float> input_value_bounds = incoming_synapse->input_neuron->value_ranges;
  std::uniform_real_distribution<float> overall_dist(input_value_bounds.first, input_value_bounds.second);
  float new_bound_center = overall_dist(gen);

  float new_bound_max_range = (fabs(input_value_bounds.first) + fabs(input_value_bounds.second)) * bound_max_range;
  std::uniform_real_distribution<float> lower_bound_dist(new_bound_center - new_bound_max_range, new_bound_center);
  std::uniform_real_distribution<float> upper_bound_dist(new_bound_center, new_bound_center + new_bound_max_range);

  this->activation_bounds[incoming_synapse->id] = std::make_pair(lower_bound_dist(gen), upper_bound_dist(gen));
}

void BoundedNeuron::update_activation_bounds(synapse * incoming_synapse, float new_bound_center) {
  // assign new bounds to the bounded unit based on the provided center (can be used for imprinting)
  // TODO The ranges of the input neurons are still used though
  std::pair<float,float> input_value_bounds = incoming_synapse->input_neuron->value_ranges;
  float new_bound_max_range = (fabs(input_value_bounds.first) + fabs(input_value_bounds.second)) * bound_max_range;
  std::uniform_real_distribution<float> lower_bound_dist(new_bound_center - new_bound_max_range, new_bound_center);
  std::uniform_real_distribution<float> upper_bound_dist(new_bound_center, new_bound_center + new_bound_max_range);

  this->activation_bounds[incoming_synapse->id] = std::make_pair(lower_bound_dist(gen), upper_bound_dist(gen));
}

float BoundedNeuron ::forward(float temp_value) {
  return temp_value;
}

float BoundedNeuron ::backward(float post_activation) {
  if (!(post_activation == 1 || post_activation == 0)){
    std::cout << post_activation << std::endl;
    std::cout << "neuron.cpp: invalid backward activation" << std::endl;
    exit(1);
  }
  if (post_activation == 1)
    return 1;
  else
    return 0;
}

void BoundedNeuron ::update_value(int time_step) {
  this->neuron_age++;

  if (this->neuron_age == this->drinking_age * 4 && !this->is_output_neuron) {
    this->is_mature = true;
  }

//  Reset our gradient_activation holder
  this->value_before_firing = 1;
  this->shadow_error_prediction_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    if (it->in_shadow_mode){
      std::cout << "neuron.cpp: shadowmode not implemented" << std::endl;
      exit(1);
    }
    it->age++;
    message_activation activation_val;
    activation_val.gradient_activation = it->input_neuron->value;
    activation_val.time = time_step - 1;
    activation_val.error_prediction_value = it->input_neuron->shadow_error_prediction;
    it->weight_assignment_past_activations.push(activation_val);

    std::pair<float, float>  it_activation_bounds = this->activation_bounds[it->id];
    //TODO think about adjusting utility prop. Maybe use neuron's instead. The utility of synapses should be equal right?
    if (it->input_neuron->value < it_activation_bounds.first || it->input_neuron->value > it_activation_bounds.second)
      this->value_before_firing = 0;
  }
}



//BoundedNeuron::BoundedNeuron(bool is_input, bool is_output, float bound_replacement_prob) : Neuron(is_input, is_output) {
//  //this->is_mature = true;
//  this->bound_replacement_prob = bound_replacement_prob;
//}
//
//void BoundedNeuron::update_activation_bounds(synapse * incoming_synapse) {
//  // find a random center point and the find random lower and upper bounds that are close to the center point.
//  if (incoming_synapse->weight != 1){
//    std::cout << "neuron.cpp: BoundedNeuron bounds are based on incoming neuron value ranges" << std::endl;
//    exit(1);
//  }
//
//  std::pair<float,float> input_value_bounds = incoming_synapse->input_neuron->value_ranges;
//  std::uniform_real_distribution<float> overall_dist(input_value_bounds.first, input_value_bounds.second);
//  float new_bound_center = overall_dist(gen);
//
//  //TODO 0.05 is hparam
//  float new_bound_max_range = (fabs(input_value_bounds.first) + fabs(input_value_bounds.second)) * 0.035;
//  std::uniform_real_distribution<float> lower_bound_dist(new_bound_center - new_bound_max_range, new_bound_center);
//  std::uniform_real_distribution<float> upper_bound_dist(new_bound_center, new_bound_center + new_bound_max_range);
//
//  this->activation_bounds[incoming_synapse->id] = std::make_pair(lower_bound_dist(gen), upper_bound_dist(gen));
//
//  //Reset the weights of the outgoing synapses
//  //TODO check what else to reset
//  for (auto &it : this->outgoing_synapses) {
//    it->weight = 1e-4;
//    it->reset_trace();
//  }
//
//  std::cout << "bound update.." << std::endl;
//}
//
//float BoundedNeuron ::forward(float temp_value) {
//  return temp_value;
//}
//
//float BoundedNeuron ::backward(float post_activation) {
//  if (!(post_activation == 1 || post_activation == 0)){
//    std::cout << post_activation << std::endl;
//    std::cout << "neuron.cpp: invalid backward activation" << std::endl;
//    exit(1);
//  }
//  if (post_activation = 1)
//    return 1;
//  else
//    return 0;
//}
//
//void BoundedNeuron ::update_value(int time_step) {
//  this->neuron_age++;
//
//  if (this->neuron_age == this->drinking_age * 4 && !this->is_output_neuron) {
//    this->is_mature = true;
//  }
//
//  Reset our gradient_activation holder
//  this->value_before_firing = 1;
//  this->shadow_error_prediction_before_firing = 0;
//
////  Age our neuron like a fine wine and set the next values of our neuron.
//  for (auto &it : this->incoming_synapses) {
//    if (it->in_shadow_mode){
//      std::cout << "neuron.cpp: shadowmode not implemented" << std::endl;
//      exit(1);
//    }
//    it->age++;
//    message_activation activation_val;
//    activation_val.gradient_activation = it->input_neuron->value;
//    activation_val.time = time_step - 1;
//    activation_val.error_prediction_value = it->input_neuron->shadow_error_prediction;
//    it->weight_assignment_past_activations.push(activation_val);
//
//    std::uniform_real_distribution<float> dist(0,1);
//    //TODO adjust weight to 00
//    //TODO synapse_utility is -nan
//    if (dist(gen) <= this->bound_replacement_prob && it->synapse_utility < it->utility_to_keep)
//      update_activation_bounds(it);
//    std::pair<float, float>  it_activation_bounds = this->activation_bounds[it->id];
//    //TODO think about adjusting utility prop. Maybe use neuron's instead. The utility of synapses should be equal right?
//    if (it->input_neuron->value < it_activation_bounds.first || it->input_neuron->value > it_activation_bounds.second)
//      this->value_before_firing = 0;
//  }
//}
//


std::mt19937 Neuron::gen = std::mt19937(0);

int64_t Neuron::neuron_id_generator = 0;
