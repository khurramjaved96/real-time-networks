#include "../../../include/nn/networks/imprinting_atari_network.h"
#include <numeric>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <random>
#include <execution>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include "../../../include/nn/neuron.h"
#include "../../../include/nn/synapse.h"
#include "../../../include/nn/dynamic_elem.h"
#include "../../../include/utils.h"
#include "../../../include/nn/utils.h"


ImprintingAtariNetwork::ImprintingAtariNetwork(int no_of_input_features,
                                               int no_of_output_neurons,
                                               int width_of_network,
                                               float step_size,
                                               float meta_step_size,
                                               bool tidbd,
                                               int seed,
                                               bool use_imprinting,
                                               int input_H,
                                               int input_W,
                                               int input_bins) {

  // TODO increment references not handled
  this->time_step = 0;
  this->mt.seed(seed);
  Neuron::gen = std::mt19937(seed);
  this->use_imprinting = use_imprinting;
  this->step_size = step_size;
  this->meta_step_size = meta_step_size;
  this->input_H = input_H;
  this->input_W = input_W;
  this->input_bins = input_bins;

  std::vector<int> inp(no_of_input_features);
  std::iota(inp.begin(), inp.end(), 0);
  this->input_indices = inp;

  std::uniform_real_distribution<float> dist(0, 1);

  for (int neuron_no = 0; neuron_no < no_of_input_features; neuron_no++) {
    auto n = new LinearNeuron(true, false);
    n->is_mature = true;
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int neuron_no = 0; neuron_no < no_of_output_neurons; neuron_no++) {
    auto n = new LinearNeuron(false, true);
    n->is_mature = true;
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  // bias unit that is always 0, just to make the output receive inputs at start
  this->bias_unit = new BiasNeuron();
  this->bias_unit->is_mature = false;
  this->bias_unit->is_bias_unit = true;
  this->all_neurons.push_back(bias_unit);

  for (auto &output : this->output_neurons) {
    synapse *s = new synapse(bias_unit, output, 0.0,0);
    s->disable_utility = true;
    s->block_gradients();
    this->all_synapses.push_back(s);
    this->output_synapses.push_back(s);
    s->set_meta_step_size(0);
  }

}



void ImprintingAtariNetwork::step() {
  //  Calculate and temporarily hold our next neuron values.
  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->update_value(this->time_step);
      });

  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->fire(this->time_step);
      });

//  Contrary to the name, this function passes gradients BACK to the incoming synapses
//  of each neuron.
  std::for_each(
      std::execution::par_unseq,
      output_neurons.begin(),
      output_neurons.end(),
      [&](Neuron *n) {
        n->forward_gradients();
      });

//  Now we propagate our error backwards one step
  std::for_each(
      std::execution::par_unseq,
      output_neurons.begin(),
      output_neurons.end(),
      [&](Neuron *n) {
        n->propagate_error();
      });

//  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      output_synapses.begin(),
      output_synapses.end(),
      [&](synapse *s) {
        s->assign_credit();
      });

//  Update our weights (based on either normal update or IDBD update
  std::for_each(
      std::execution::par_unseq,
      output_synapses.begin(),
      output_synapses.end(),
      [&](synapse *s) {
        s->update_weight();
      });

 //Mark all is_useless weights and neurons for deletion
  std::for_each(
      std::execution::par_unseq,
      imprinted_features.begin(),
      imprinted_features.end(),
      [&](Neuron *n) {
        n->mark_useless_weights();
      });

//  Delete our is_useless weights and neurons
  std::for_each(
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->prune_useless_weights();
      });

  // TODO no garbage collection here for now
  auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
  this->all_synapses.erase(it, this->all_synapses.end());

  it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_s);
  this->output_synapses.erase(it, this->output_synapses.end());

  auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
  this->all_neurons.erase(it_n, this->all_neurons.end());

  it_n = std::remove_if(this->imprinted_features.begin(), this->imprinted_features.end(), to_delete_n);
  this->imprinted_features.erase(it_n, this->imprinted_features.end());

  this->time_step++;
}

void ImprintingAtariNetwork::imprint_LTU_randomly() {
  std::uniform_real_distribution<float> prob_sampler(0, 0.5);
  float percentage_to_look_at = prob_sampler(this->mt);

  std::vector <Neuron *> interesting_neurons;
  for (auto &it : this->all_neurons) {
    if (it->value != it->old_value && it->is_mature && !it->useless_neuron && !it->is_output_neuron && it->is_input_neuron)
      interesting_neurons.push_back(it);
  }

  int total_ones = 0;
  auto new_feature = new LTU(false, false, 100000);
  for (auto &it : interesting_neurons){
    if(prob_sampler(this->mt) > percentage_to_look_at) {
      auto s = new synapse(it, new_feature, 1, 0);
      this->all_synapses.push_back(s);
      s->set_meta_step_size(0);
      total_ones++;
    }
  }

  if (total_ones != 0){
    this->all_neurons.push_back(new_feature);
    this->imprinted_features.push_back(new_feature);
    std::uniform_real_distribution<float> thres_sampler(0, total_ones);
    new_feature->activation_threshold = thres_sampler(this->mt);

    //This blows up :/
    //float imprinting_weight = -1 * this->output_neurons[0]->error_gradient.back().error;
    float imprinting_weight = 0.01 * prob_sampler(this->mt);
    auto s = new synapse(new_feature, this->output_neurons[0], imprinting_weight, this->step_size);
    this->all_synapses.push_back(s);
    this->output_synapses.push_back(s);
    s->set_meta_step_size(this->meta_step_size);
    s->turn_on_idbd();
    s->block_gradients(); //TODO will this affect utility prop?
    std::cout << "Interesting: " << interesting_neurons.size() << " selected: " << total_ones << " thresh: " << new_feature->activation_threshold << " weight: " << imprinting_weight << " total current features: " << this->imprinted_features.size() << " total syn: " << this->all_synapses.size() << std::endl;

  }
  else
    delete new_feature;
}



//void ImprintingAtariNetwork::set_input_values(std::vector<float> const &input_values) {
////    assert(input_values.size() == this->input_neurons.size());
//  for (int i = 0; i < input_values.size(); i++) {
//    if (i < this->input_neurons.size()) {
//      this->input_neurons[i]->old_value = this->input_neurons[i]->value;
//      this->input_neurons[i]->old_value_without_activation = this->input_neurons[i]->value;
//      this->input_neurons[i]->value = input_values[i];
//      this->input_neurons[i]->value_without_activation = input_values[i];
//    } else {
//      std::cout << "More input features than input neurons\n";
//      exit(1);
//    }
//  }
//}

void ImprintingAtariNetwork::set_input_values(std::vector<float> const &input_values) {
  if (input_values.size() != this->input_neurons.size()){
    std::cout << "err size in set_input_values()" << std::endl;
    exit(1);
  }

  std::for_each(
      std::execution::par_unseq,
      this->input_indices.begin(),
      this->input_indices.end(),
      [&](int i) {
        this->input_neurons[i]->old_value = this->input_neurons[i]->value;
        this->input_neurons[i]->old_value_without_activation = this->input_neurons[i]->value;
        this->input_neurons[i]->value = input_values[i];
        this->input_neurons[i]->value_without_activation = input_values[i];
      });

//  for (int i = 0; i < input_values.size(); i++) {
//    if (i < this->input_neurons.size()) {
//      if (this->input_neurons[i]->value != input_values[i] or this->input_neurons[i]->value_without_activation != input_values[i]){
//        std::cout << "bugged!~" << std::endl;
//        exit(1);
//      }
//    }
//  }
}
