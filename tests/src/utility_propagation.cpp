//
// Created by Khurram Javed on 2021-08-16.
//

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <random>

#include "../include/test_cases.h"
#include "../../include/utils.h"
#include "../../include/environments/mountain_car.h"
#include "../../include/nn/networks/linear_function_approximator.h"
#include "../include/test_case_networks.h"
#include "../../include/nn/utils.h"

bool utility_test() {
  TestCase my_network = TestCase();


  auto a = new LinearNeuron(true, false);
  my_network.all_neurons.push_back(a);
  my_network.input_neurons.push_back(a);

  auto b = new LinearNeuron(true, false);
  my_network.all_neurons.push_back(b);
  my_network.input_neurons.push_back(b);

  auto c = new LinearNeuron(true, false);
  my_network.all_neurons.push_back(c);
  my_network.input_neurons.push_back(c);

//

  auto d = new ReluNeuron(false, false);
  my_network.all_neurons.push_back(d);


  auto e = new ReluNeuron(false, false);
  my_network.all_neurons.push_back(e);

  auto f = new SigmoidNeuron(false, false);
  my_network.all_neurons.push_back(f);

  auto g = new ReluNeuron(false, false);
  my_network.all_neurons.push_back(g);


  auto h = new LinearNeuron(false, true);
  my_network.all_neurons.push_back(h);
  my_network.output_neurons.push_back(h);

  my_network.all_synapses.push_back(new synapse(a, d, 0.1, 0));
  my_network.all_synapses.push_back(new synapse(d, g, 0.5, 0));
  my_network.all_synapses.push_back(new synapse(g, h, 2, 0));
  my_network.all_synapses.push_back(new synapse(d, h, 3, 0));
  my_network.all_synapses.push_back(new synapse(a, h, 0.5, 0));
  my_network.all_synapses.push_back(new synapse(b, e, 4, 0));
  my_network.all_synapses.push_back(new synapse(e, h, 0.4, 0));
  my_network.all_synapses.push_back(new synapse(c, f, 1, 0));
  my_network.all_synapses.push_back(new synapse(f, h, 2, 0));
  my_network.all_synapses.push_back(new synapse(e, f, 2, 0));
  my_network.all_synapses.push_back(new synapse(c, h, 4, 0));

  float output_val = 0;
  std::vector<float> inp;
  for (int i = 0; i < my_network.input_neurons.size(); i++) {
    inp.push_back(1);
  }
//  inp.push_back(0.1);
  for (int i = 0; i < 20000; i++) {

    my_network.set_input_values(inp);
    my_network.step();

    my_network.introduce_targets(my_network.read_output_values(), 0, 0);
    output_val = my_network.read_output_values()[0];
//    if(i == 5000){
//      std::cout << "ID\tUtility\tUtilityToD\n";
//      for (auto neuron_it: my_network.all_neurons) {
//        std::cout << neuron_it->id << "\t"  << neuron_it->neuron_utility  << std::endl;
//      }
//
//      std::cout << "Utility of syanpse at step " << std::endl;
//      std::cout << "From\tTo\tWeight\tUtility\tUtilToD\n";
//      for (auto synapse_it: my_network.all_synapses) {
//        std::cout << synapse_it->input_neuron->id << "\t" << synapse_it->output_neuron->id << "\t" << synapse_it->weight
//                  << "\t" << synapse_it->synapse_utility << "\t" << synapse_it->synapse_utility_to_distribute << std::endl;
//      }
//    }
  }
//  std::cout << "ID\tUtility\tUtilityToD\n";
//  for (auto neuron_it: my_network.all_neurons) {
//    std::cout << neuron_it->id << "\t" <<  neuron_it->neuron_utility  << std::endl;
//  }
//
//  std::cout << "Utility of syanpse at step " << std::endl;
//  std::cout << "From\tTo\tWeight\tUtility\tUtilToD\n";
//  for (auto synapse_it: my_network.all_synapses) {
//    std::cout << synapse_it->input_neuron->id << "\t" << synapse_it->output_neuron->id << "\t" << synapse_it->weight
//              << "\t" << synapse_it->synapse_utility << "\t" << synapse_it->synapse_utility_to_distribute << std::endl;
//  }
//
//  std::cout << "Output val = " << output_val << std::endl;

  return true;
}