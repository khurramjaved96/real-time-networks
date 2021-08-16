//
// Created by Khurram Javed on 2021-08-12.
//

#include "../../../include/nn/networks/linear_function_approximator.h"
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


LinearFunctionApproximator::LinearFunctionApproximator(int no_of_input_features,
                                                       int no_of_output_neurons,
                                                       float step_size,
                                                       float meta_step_size,
                                                       bool tidbd) {
  this->time_step = 0;
  for (int neuron_no = 0; neuron_no < no_of_input_features; neuron_no++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int neuron_no = 0; neuron_no < no_of_output_neurons; neuron_no++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int inp_neuron = 0; inp_neuron < no_of_input_features; inp_neuron++) {
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

void LinearFunctionApproximator::step() {
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


//void LinearFunctionApproximator::introduce_targets(std::vector<float> targets) {
//  std::cout << "Use introduce target with lambda and gamma interface\n";
//  exit(1);
//}
//
