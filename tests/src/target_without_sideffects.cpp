//
// Created by Khurram Javed on 2021-08-05.
//


#include "../include/test_cases.h"

#include <iostream>
#include <vector>

#include "../include/test_case_networks.h"
#include "../../include/utils.h"
#include "../../include/environments/animal_learning/tracecondioning.h"


bool forward_pass_without_sideeffects_test(){
  ForwardPassWithoutSideEffects my_network = ForwardPassWithoutSideEffects();
  ForwardPassWithoutSideEffects my_network_2 = ForwardPassWithoutSideEffects();
  long long int time_step = 0;
//  std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 100; a++) {
    std::vector<float> curr_inp;

    if (a < 100) {
      curr_inp.push_back((a - 0) * 0.01);
      curr_inp.push_back(10 - ((a - 0) * 0.1));
      curr_inp.push_back(a - 0);
    }
    input_list.push_back(curr_inp);
  }
  int counter = 0;
  float sum_of_activation = 0;
  for (auto it : input_list) {
    float prediction = my_network.forward_pass_without_side_effects(it)[0];
    my_network.set_input_values(it);
    my_network_2.set_input_values(it);
    my_network.step();
    my_network_2.step();
    std::vector<float> output = my_network.read_output_values();
    std::vector<float> output_2 = my_network_2.read_output_values();
    if(output[0] != output_2[0] || output[0] != prediction){
      return false;
    }
//    std::cout  << "Actual prediction: " << output[0] << " Output 2 " << output_2[0] << " Predicted: " << prediction << std::endl;

    my_network.introduce_targets(output);
    my_network_2.introduce_targets(output);
    counter++;
  }

  return true;
}

ForwardPassWithoutSideEffects::ForwardPassWithoutSideEffects() {
  this->time_step = 0;

  int input_neuron = 3;
  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  bool relu = true;
  auto n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  float step_size = 0;
  auto inp1 = new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size);
  auto inp2 = new synapse(all_neurons[1 - 1], all_neurons[7 - 1], 0.5, step_size);
  inp1->block_gradients();
  inp2->block_gradients();
  this->all_synapses.push_back(inp1);
  this->all_synapses.push_back(inp2);
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[5 - 1], 0.7, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[5 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[7 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[6 - 1], all_neurons[7 - 1], 0.2, step_size));

}
