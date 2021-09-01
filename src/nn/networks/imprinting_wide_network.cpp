#include "../../../include/nn/networks/imprinting_wide_network.h"
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


ImprintingWideNetwork::ImprintingWideNetwork(int no_of_input_features,
                                             int no_of_output_neurons,
                                             int width_of_network,
                                             std::vector<std::pair<float,float>> input_ranges,
                                             float bound_replacement_prob,
                                             float step_size,
                                             float meta_step_size,
                                             bool tidbd) {


  //TODO provide seed
  this->time_step = 0;
  this->bound_replacement_prob = bound_replacement_prob;
  if (input_ranges.size() != no_of_input_features){
    std::cout << "input_ranges shape should be equal to no_of_input_features" << std::endl;
    exit(1);
  }

  for (int neuron_no = 0; neuron_no < no_of_input_features; neuron_no++) {
    if (input_ranges[neuron_no].first >= input_ranges[neuron_no].second){
      std::cout << "input_ranges should be [low,high] rather than [high,low]"<< std::endl;
      exit(1);
    }
  }

  for (int neuron_no = 0; neuron_no < no_of_input_features; neuron_no++) {
    auto n = new LinearNeuron(true, false);
    n->value_ranges = std::make_pair(input_ranges[neuron_no].first, input_ranges[neuron_no].second);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int neuron_no = 0; neuron_no < no_of_output_neurons; neuron_no++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int neuron_no = 0; neuron_no < width_of_network; neuron_no++) {
    auto n = new BoundedNeuron(false, false, this->bound_replacement_prob);
    this->bounded_neurons.push_back(n);
    this->all_neurons.push_back(n);
    for (int inp_neuron = 0; inp_neuron < no_of_input_features; inp_neuron++) {
      auto s = new synapse(this->input_neurons[inp_neuron], n, 1, 0);
      this->all_synapses.push_back(s);
      n->update_activation_bounds(s);
    }
    for (int out_neuron = 0; out_neuron < no_of_output_neurons; out_neuron++) {
      auto s = new synapse(n, this->output_neurons[out_neuron], 1e-4, step_size);
      this->output_synapses.push_back(s);
      this->all_synapses.push_back(s);
      s->set_meta_step_size(meta_step_size);
      s->block_gradients();
      if (tidbd)
        s->turn_on_idbd();
    }
  }
}

std::vector<std::vector<std::pair<float, float>>> ImprintingWideNetwork::get_feature_bounds(){
  std::vector<std::vector<std::pair<float,float>>> all_bounds;
  for (auto neuron_it : this->bounded_neurons){
    std::vector<std::pair<float,float>> neuron_bounds;
    for (auto syn_it : neuron_it->incoming_synapses)
      neuron_bounds.push_back(neuron_it->activation_bounds[syn_it->id]);
    all_bounds.push_back(neuron_bounds);
  }
  return all_bounds;
}

std::vector<float> ImprintingWideNetwork::get_feature_utilities(){
  std::vector<float> feature_utilities;
  for (auto neuron_it : this->bounded_neurons)
    feature_utilities.push_back(neuron_it->neuron_utility);
  return feature_utilities;
}


void ImprintingWideNetwork::step() {
  //  Calculate and temporarily hold our next neuron values.
  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        BoundedNeuron *ptr = dynamic_cast<BoundedNeuron*>(n);
        if (ptr)
          ptr->update_value(this->time_step);
        else
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
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->forward_gradients();
      });

//  Now we propagate our error backwards one step
  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->propagate_error();
      });

//  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->assign_credit();
      });

//  Update our weights (based on either normal update or IDBD update
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->update_weight();
      });

  this->time_step++;
}
