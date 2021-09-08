#include "../../../include/nn/networks/expanding_linear_function_approximator.h"
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


ExpandingLinearFunctionApproximator::ExpandingLinearFunctionApproximator(int no_of_input_features,
                                                                         int no_of_output_neurons,
                                                                         int no_of_active_input_features,
                                                                         float step_size,
                                                                         float meta_step_size,
                                                                         bool tidbd) {
  this->time_step = 0;
  this->no_of_active_input_features = no_of_active_input_features;
  this->step_size = step_size;
  this->meta_step_size = meta_step_size;
  this->tidbd = tidbd;

  for (int neuron_no = 0; neuron_no < no_of_input_features; neuron_no++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    if (neuron_no < no_of_active_input_features)
      this->all_neurons.push_back(n);
  }

  for (int neuron_no = 0; neuron_no < no_of_output_neurons; neuron_no++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int inp_neuron = 0; inp_neuron < no_of_active_input_features; inp_neuron++) {
    for (int out_neuron = 0; out_neuron < no_of_output_neurons; out_neuron++) {
      auto s = new synapse(this->input_neurons[inp_neuron], this->output_neurons[out_neuron], 0, step_size);
      this->output_synapses.push_back(s);
      this->all_synapses.push_back(s);
      s->set_meta_step_size(meta_step_size);
      if (tidbd) {
        s->turn_on_idbd();
      }
    }
  }

}

void ExpandingLinearFunctionApproximator::set_input_values(std::vector<float> const &input_values) {
  if (input_values.size() > this->no_of_active_input_features){
    for (int neuron_no = this->no_of_active_input_features; neuron_no < input_values.size(); neuron_no++){
      this->no_of_active_input_features+=1;
      //std::cout << "activating input neuron no: " << this->no_of_active_input_features << std::endl;
      this->all_neurons.push_back(this->input_neurons[neuron_no]);

      for (int out_neuron = 0; out_neuron < this->output_neurons.size(); out_neuron++) {
        auto s = new synapse(this->input_neurons[neuron_no], this->output_neurons[out_neuron], 0, this->step_size);
        this->output_synapses.push_back(s);
        this->all_synapses.push_back(s);
        s->set_meta_step_size(this->meta_step_size);
        if (this->tidbd) {
          s->turn_on_idbd();
        }
      }


    }
  }
  for (int i = 0; i < input_values.size(); i++) {
    if (i < this->input_neurons.size()) {
      this->input_neurons[i]->value = input_values[i];
    } else {
      std::cout << "More input features than input neurons\n";
      exit(1);
    }
  }
}

void ExpandingLinearFunctionApproximator::step() {
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


