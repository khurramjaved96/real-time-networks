//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//


#include "../../../include/nn/networks/feedforward_state_value_network.h"
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


ContinuallyAdaptingNetwork::ContinuallyAdaptingNetwork(float step_size, int seed,
                                                       int no_of_input_features, int no_of_output_features) {
  this->time_step = 0;
  this->mt.seed(seed);
//  Initialize the neural network input neurons.
//  Currently we fix an input size of 10.
  int input_neuron = no_of_input_features;

  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    n->is_mature = true;
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
    n->increment_reference();
    this->input_neurons.push_back(n);
    n->increment_reference();
    this->all_neurons.push_back(n);
  }

//  Initialize all output neurons.
//  Similarly, we fix an output size to 1.

  int output_neuros = no_of_output_features;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
    n->increment_reference();
    this->output_neurons.push_back(n);
    n->increment_reference();
    this->all_neurons.push_back(n);
  }


//  Connect our input and output neurons with synapses.
  for (auto &input : this->input_neurons) {
    for (auto &output : this->output_neurons) {
      synapse *s = new synapse(input, output, 0, step_size);
      this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
      s->increment_reference();
      this->all_synapses.push_back(s);
      s->increment_reference();
      this->output_synapses.push_back(s);
      s->turn_on_idbd();
    }
  }
}

void ContinuallyAdaptingNetwork::print_graph(Neuron *root) {
  for (auto &os : root->outgoing_synapses) {
    auto current_n = os;

    if (!current_n->print_status) {
      std::cout << current_n->input_neuron->id << "\t" << current_n->output_neuron->id << "\t"
                << os->grad_queue.size() << "\t\t" << current_n->input_neuron->past_activations.size()
                << "\t\t\t" << current_n->output_neuron->past_activations.size() << "\t\t\t"
                << current_n->input_neuron->error_gradient.size()
                << "\t\t" << current_n->credit << std::endl;
      current_n->print_status = true;
    }
    print_graph(current_n->output_neuron);
  }
}

void ContinuallyAdaptingNetwork::viz_graph() {
  NetworkVisualizer netviz = NetworkVisualizer(this->all_neurons);
  netviz.generate_dot(this->time_step);
  netviz.generate_dot_detailed(this->time_step);
}

std::string ContinuallyAdaptingNetwork::get_viz_graph() {
  NetworkVisualizer netviz = NetworkVisualizer(this->all_neurons);
  return netviz.get_graph(this->time_step);
//    netviz.generate_dot_detailed(this->time_step);
}

/**
 * Add a feature by adding a neuron to the neural network. This neuron is connected
 * to each (non-output) neuron w.p. perc ~ U(0, 1) and connected to each output neuron
 * with either a -1 and 1 weight.
 * @param step_size: step size of the input synapse added. Step size of the output synapse added starts as 0.
 */
void ContinuallyAdaptingNetwork::add_feature(float step_size) {
//  Limit our number of synapses to 1m
  if (this->all_synapses.size() < 1000000) {
//        std::normal_distribution<float> dist(0, 1);
    std::uniform_int_distribution<int> drinking_dist(5000, 10000);
    std::uniform_real_distribution<float> dist(-2, 2);
    std::uniform_real_distribution<float> dist_u(0, 1);
    std::uniform_real_distribution<float> dist_recurren(0, 0.99);

//      Create our new neuron
    Neuron *new_feature = new SigmoidNeuron(false, false);
//        recurrent_neuron->drinking_age = drinking_dist(this->mt);
    new_feature->increment_reference();
    new_feature->increment_reference();
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(new_feature));
    this->all_neurons.push_back(new_feature);


    Neuron *bias_unit = new BiasNeuron();
    bias_unit->increment_reference();
    bias_unit->increment_reference();
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(bias_unit));
    this->all_neurons.push_back(bias_unit);

    auto bias_syanpse = new synapse(bias_unit, new_feature, 0.1 * dist(this->mt), step_size);
    bias_syanpse->block_gradients();
    bias_syanpse->increment_reference();
    this->all_synapses.push_back(bias_syanpse);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(bias_syanpse));

//      w.p. perc, attach a random neuron (that's not an output neuron) to this neuron
    float perc = dist_u(mt)*0.1;
    for (auto &n : this->all_neurons) {
      if (!n->is_output_neuron && n->is_mature) {
        if (dist_u(mt) < perc) {
          auto syn = new synapse(n, new_feature, 0.1 * dist(this->mt), step_size);
          syn->block_gradients();
          syn->increment_reference();
          this->all_synapses.push_back(syn);
          this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
        }
      }
    }

    synapse *output_s_temp;
    if (dist(this->mt) > 0) {
      output_s_temp = new synapse(new_feature, this->output_neurons[0], 0, step_size);
    } else {
      output_s_temp = new synapse(new_feature, this->output_neurons[0], 0, step_size);
    }
//    output_s_temp->set_shadow_weight(true);
    output_s_temp->turn_on_idbd();
    output_s_temp->increment_reference();
    this->all_synapses.push_back(output_s_temp);
    output_s_temp->increment_reference();
    this->output_synapses.push_back(output_s_temp);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_s_temp));
  }
}

void ContinuallyAdaptingNetwork::set_print_bool() {
  std::cout
      << "From\tTo\tGrad_queue_size\tFrom activations_size\tTo activations_size\tError grad_queue From\tCredit\n";
  for (auto &s : this->all_synapses)
    s->print_status = false;
}

ContinuallyAdaptingNetwork::~ContinuallyAdaptingNetwork() {
  for (auto &it : this->all_heap_elements)
    delete it;
}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda) {
//  Put all targets into our neurons.
  float error = 0;
  if (targets.size() != 1) {
    std::cout << "Multiple target values passed. This network only learns to make a single prediction.\n";
    exit(1);
  }
  for (int counter = 0; counter < targets.size(); counter++) {
    error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1, gamma, lambda);
  }
  return error * error;
}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets,
                                                    float gamma,
                                                    float lambda,
                                                    std::vector<bool> no_grad) {
  float error = 0;
//  if (targets.size() != 1) {
//    std::cout << "Multiple target values passed. This network only learns to make a single prediction.\n";
//    exit(1);
//  }
  for (int counter = 0; counter < targets.size(); counter++) {
    error += this->output_neurons[counter]->introduce_targets(targets[counter],
                                                              this->time_step - 1,
                                                              gamma,
                                                              lambda,
                                                              no_grad[counter]);
  }
  return error * error;
}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets) {
  std::cout << "Interface not supported for this networm. Please use one with gamme and lambda values\n";
  exit(1);
}

