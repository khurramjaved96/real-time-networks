#include "../../../include/nn/networks/synced_network.h"

#include <assert.h>
#include <cmath>
#include <random>
#include <execution>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include "../../../include/nn/synced_neuron.h"
#include "../../../include/nn/synced_synapse.h"
#include "../../../include/nn/dynamic_elem.h"
#include "../../../include/utils.h"
#include "../../../include/nn/utils.h"

/**
 * Continually adapting neural network.
 * Essentially a neural network with the ability to add and remove neurons
 * based on a generate and test approach.
 * Check the corresponding header file for a description of the variables.
 *
 * As a quick note as to how this NN works - it essentially fires all neurons once
 * per step, unlike a usual NN that does a full forward pass per output needed.
 *
 * @param step_size: neural network step size.
 * @param width: [NOT CURRENTLY USED] neural network width
 * @param seed: random seed to initialize.
 */

int SyncedNetwork::get_total_neurons() {
  int tot = 0;
  for (auto it : this->all_neurons) {
    if (it->is_mature)
      tot++;
  }
  return tot;
}

SyncedNetwork::SyncedNetwork() {
  this->time_step = 0;
}

int64_t SyncedNetwork::get_timestep() {
  return this->time_step;
}

int SyncedNetwork::get_input_size() {
  return this->input_neurons.size();
}

int SyncedNetwork::get_total_synapses() {
  int tot = 0;
  for (auto it : this->all_synapses) {
    if (it->output_neuron->is_mature && it->input_neuron->is_mature)
      tot++;
  }
  return tot;
}

SyncedNetwork::~SyncedNetwork() {
  for (auto &it : this->all_heap_elements)
    delete it;
}

void SyncedNetwork::set_input_values(std::vector<float> const &input_values) {
//    assert(input_values.size() == this->input_neurons.size());
  for (int i = 0; i < input_values.size(); i++) {
    if (i < this->input_neurons.size()) {
      this->input_neurons[i]->old_value = this->input_neurons[i]->value;
      this->input_neurons[i]->old_value_without_activation = this->input_neurons[i]->value;
      this->input_neurons[i]->value = input_values[i];
      this->input_neurons[i]->value_without_activation = input_values[i];
    } else {
      std::cout << "More input features than input neurons\n";
      exit(1);
    }
  }
}


void SyncedNetwork::print_neuron_status() {
  std::cout << "ID\tUtility\tAvg activation\n";
  for(auto it : this->all_neurons){
    if(it->is_mature) {
      std::cout << it->id << "\t" << it->neuron_utility << "\t\t" << it->average_activation << std::endl;
    }
  }
}

void SyncedNetwork::print_synapse_status() {
  std::cout << "From\tTo\tWeight\tUtil\tUtiltoD\tStep-size\tAge\n";
  for(auto it : this->all_synapses){
    if(it->output_neuron->is_mature && it->input_neuron->is_mature) {
      std::cout << it->input_neuron->id << "\t" << it->output_neuron->id << "\t" << it->weight << "\t"
                << it->synapse_utility << "\t" << it->synapse_utility_to_distribute << "\t" << it->step_size << "\t"
                << it->age << std::endl;
    }
  }
}

/**
 * Step function after putting in the inputs to the neural network.
 * This function takes a step in the NN by firing all neurons.
 * Afterwards, it calculates gradients based on previous error and
 * propagates it back. Currently backprop is truncated at 1 step.
 * Finally, it updates its weights and prunes is_useless neurons and synapses.
 */

void SyncedNetwork::step() {

}

/**
 * Find all synapses and neurons with 0 references to them and delete them.
 */
void SyncedNetwork::collect_garbage() {
  for (int temp = 0; temp < this->all_heap_elements.size(); temp++) {
    if (all_heap_elements[temp]->references == 0) {
      delete all_heap_elements[temp];
      all_heap_elements[temp] = nullptr;
    }
  }

  auto it = std::remove_if(this->all_heap_elements.begin(), this->all_heap_elements.end(), is_null_ptr);
  this->all_heap_elements.erase(it, this->all_heap_elements.end());
}

std::vector<float> SyncedNetwork::read_output_values() {
  std::vector<float> output_vec;
  output_vec.reserve(this->output_neurons.size());
  for (auto &output_neuro : this->output_neurons) {
    output_vec.push_back(output_neuro->value);
  }
  return output_vec;
}

std::vector<float> SyncedNetwork::read_all_values() {
  std::vector<float> output_vec;
  output_vec.reserve(this->all_neurons.size());
  for (auto &output_neuro : this->all_neurons) {
    output_vec.push_back(output_neuro->value);
  }
  return output_vec;
}

// With this interface, step-size adaptation should only be done for the outgoing prediction weights.
// For step-size adaptation for preceeding weights, user must use intordue targets with gamma and lambda
// (pass gamma = lambda = 0 in-case traces are not needed).

float SyncedNetwork::introduce_targets(std::vector<float> targets) {
  float error = 0;
  for (int counter = 0; counter < targets.size(); counter++) {
    error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step);
  }
  return error;
}

std::vector<float> SyncedNetwork::forward_pass_without_side_effects(std::vector<float> input_values) {

  std::vector<float> backup_values;
  backup_values.reserve(this->input_neurons.size());
  for (int i = 0; i < input_values.size(); i++) {
    if (i < this->input_neurons.size()) {
      backup_values.push_back(this->input_neurons[i]->value);
      this->input_neurons[i]->value = input_values[i];
    } else {
      std::cout << "More input features than input neurons\n";
      exit(1);
    }
  }
  std::vector<float> results;
  for (auto n : this->output_neurons) {
    float temp_value = 0;
    for (auto it: n->incoming_synapses) {
      if (it->in_shadow_mode) {
//        this->shadow_error_prediction_before_firing += it->weight * it->input_neuron->value;
      } else {
        temp_value += it->weight * it->input_neuron->value;
      }
    }
    results.push_back(n->forward(temp_value));
  }
  for (int i = 0; i < backup_values.size(); i++) {

    this->input_neurons[i]->value = backup_values[i];

  }
  return results;
}

float SyncedNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda) {
//  Put all targets into our neurons.
  float error = 0;
  for (int counter = 0; counter < targets.size(); counter++) {
    error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step, gamma, lambda);
  }
  return error * error;
}


void SyncedNetwork::reset_trace() {
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](SyncedSynapse *s) {
        s->reset_trace();
      });
}

