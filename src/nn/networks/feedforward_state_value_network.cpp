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


ContinuallyAdaptingNetwork::ContinuallyAdaptingNetwork(float step_size, int seed, int no_of_input_features, float utility_to_keep = 0.001) {
  this->time_step = 0;
  this->mt.seed(seed);

  this->bias_unit = new BiasNeuron();
  this->bias_unit->is_mature = true;
  bias_unit->increment_reference();
  bias_unit->increment_reference();
  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(bias_unit));
  this->all_neurons.push_back(bias_unit);


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

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    n->is_mature = true;
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
    n->increment_reference();
    this->output_neurons.push_back(n);
    n->increment_reference();
    this->all_neurons.push_back(n);
  }


//  for (auto &output : this->output_neurons) {
//    synapse *s = new synapse(bias_unit, output, 0, step_size);
//    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
//    s->increment_reference();
//    this->all_synapses.push_back(s);
//    s->increment_reference();
//    this->output_synapses.push_back(s);
//    s->turn_on_idbd();
//    s->set_meta_step_size(1e-3);
//  }
//
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
      s->set_utility_to_keep(utility_to_keep);
      s->set_meta_step_size(1e-2);
    }
  }
}
//
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

void ContinuallyAdaptingNetwork::add_feature_binary(float step_size, float utility_to_keep) {
  if (this->all_synapses.size() < 1000000) {
//        std::normal_distribution<float> dist(0, 1);
    std::uniform_int_distribution<int> drinking_dist(1000, 40000);
    std::uniform_int_distribution<int> random_int(0, this->all_neurons.size()-1);
    std::uniform_int_distribution<int> max_features(1, 50);
    std::uniform_real_distribution<float> dist(-3, 3);
    std::uniform_real_distribution<float> dist_u(0, 1);
//    std::uniform_int_distribution<int> dist_u(0, 1);
    std::uniform_real_distribution<float> dist_recurren(0, 0.99);

//      Create our new neuron
    Neuron *new_feature = new LTU(false, false, 0);
    LTU *ref = dynamic_cast<LTU *>(new_feature);
    new_feature->drinking_age = drinking_dist(this->mt);
    new_feature->increment_reference();
    new_feature->increment_reference();
//


//    std::cout << "Making new feature\n";
    float try_counter = 0;
    float total_features_selected = 0;
    int features_to_add = max_features(mt);
    std::vector<int> pos_added;
    while (true) {
      try_counter++;
      if (try_counter == 50 or total_features_selected == features_to_add)
        break;
      int pos = random_int(mt);
//      std::cout << "Pos = " << pos << std::endl;
      if (this->all_neurons[pos]->is_mature && !this->all_neurons[pos]->is_output_neuron
          && this->all_neurons[pos]->neuron_utility >= utility_to_keep) {
        if (this->all_neurons[pos]->value > 0) {
          if (std::count(pos_added.begin(), pos_added.end(), pos) == 0) {
            pos_added.push_back(pos);
            auto syn = new synapse(this->all_neurons[pos], new_feature, 1, 0);
            syn->block_gradients();
            syn->increment_reference();
            syn->set_utility_to_keep(utility_to_keep);
            this->all_synapses.push_back(syn);
            this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
            total_features_selected++;
          }
        }
      }
    }
    if (total_features_selected > 0) {
      this->all_heap_elements.push_back(static_cast<dynamic_elem *>(new_feature));
      this->all_neurons.push_back(new_feature);
      ref->activation_threshold = dist_u(this->mt) * total_features_selected;
//      std::cout << "Activation threshold = " << ref->activation_threshold << std::endl;
//      std::cout << "Total features selected = " << total_features_selected << std::endl;

      synapse *output_s_temp;
      output_s_temp = new synapse(new_feature, this->output_neurons[0], 0, step_size);
      output_s_temp->set_shadow_weight(false);
      output_s_temp->turn_on_idbd();
      output_s_temp->set_meta_step_size(3e-3);
      output_s_temp->increment_reference();
      output_s_temp->set_utility_to_keep(utility_to_keep);
      output_s_temp->block_gradients();

      this->all_synapses.push_back(output_s_temp);
      output_s_temp->increment_reference();
      this->output_synapses.push_back(output_s_temp);
      this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_s_temp));
    }
  }
}

/**
 * Add a feature by adding a neuron to the neural network. This neuron is connected
 * to each (non-output) neuron w.p. perc ~ U(0, 1) and connected to each output neuron
 * with either a -1 and 1 weight.
 * @param step_size: step size of the input synapse added. Step size of the output synapse added starts as 0.
 */

void ContinuallyAdaptingNetwork::add_feature(float step_size, float utility_to_keep = 0.001) {
//  Limit our number of synapses to 1m
  if (this->all_synapses.size() < 1000000) {
//        std::normal_distribution<float> dist(0, 1);
    std::uniform_int_distribution<int> drinking_dist(1000, 40000);
    std::uniform_int_distribution<int> random_int(0, this->all_neurons.size());
    std::uniform_int_distribution<int> max_features(1, 5);
    std::uniform_real_distribution<float> dist(-3, 3);
    std::uniform_real_distribution<float> dist_u(0, 1);
//    std::uniform_int_distribution<int> dist_u(0, 1);
    std::uniform_real_distribution<float> dist_recurren(0, 0.99);

//      Create our new neuron
    Neuron *new_feature = new ReluNeuron(false, false);
    new_feature->drinking_age = drinking_dist(this->mt) ;
//    new_feature->drinking_age = 1;
    new_feature->increment_reference();
    new_feature->increment_reference();


    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(new_feature));
    this->all_neurons.push_back(new_feature);

//
//
//    auto bias_syanpse = new synapse(this->bias_unit, new_feature,  dist(this->mt)*0.1, step_size);
//    bias_syanpse->turn_on_idbd();
//    bias_syanpse->set_meta_step_size(1e-3);
//    bias_syanpse->block_gradients();
//    bias_syanpse->increment_reference();
//    bias_syanpse->set_utility_to_keep(utility_to_keep);
//    this->all_synapses.push_back(bias_syanpse);
//    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(bias_syanpse));

//      w.p. perc, attach a random neuron (that's not an output neuron) to this neuron
    float perc = dist_u(mt);
    float try_counter = 0;
    float total_features_selected = 0;
    int features_to_add = max_features(mt);
//    features_to_add = 3;
    std::vector<int> pos_added;
    while(true){
      try_counter ++;
      if(try_counter == 1000 or total_features_selected == features_to_add)
        break;
      int pos = random_int(mt);
//      std::cout << "Pos = " << pos << std::endl;
      if(this->all_neurons[pos]->is_mature && !this->all_neurons[pos]->is_output_neuron && this->all_neurons[pos]->neuron_utility >= utility_to_keep){
        if(std::count(pos_added.begin(), pos_added.end(), pos) == 0) {
          pos_added.push_back(pos);
          auto syn = new synapse(this->all_neurons[pos], new_feature,   dist(mt)*0.001, 1e-3);
          syn->block_gradients();
//          syn->turn_on_idbd();
//          syn->set_meta_step_size(1e-3);
          syn->increment_reference();
          syn->set_utility_to_keep(utility_to_keep);
          this->all_synapses.push_back(syn);
          this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
          total_features_selected ++;
        }
      }
    }
//    std::cout << "Total features added " << total_features_selected << std::endl;
//    int counter = 0;
//    for (auto &n : this->all_neurons) {
//      if(n->is_mature && !n->is_output_neuron){
//        if (dist_u(mt) < perc or true) {
//          auto syn = new synapse(n, new_feature, 0.01 * dist(this->mt), 3e-3);
//          syn->block_gradients();
//          syn->turn_on_idbd();
//          syn->set_meta_step_size(1e-3);
//          syn->increment_reference();
//          this->all_synapses.push_back(syn);
//          this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
//        }
//        counter++;
//        if(counter == 6) break;
//      }
//    }

    synapse *output_s_temp;
    if (dist(this->mt) > 0) {
      output_s_temp = new synapse(new_feature, this->output_neurons[0], 1, 0);
    } else {
      output_s_temp = new synapse(new_feature, this->output_neurons[0], -1, 0);
    }
    output_s_temp->set_shadow_weight(false);
//    output_s_temp->turn_off_idbd();
    output_s_temp->turn_on_idbd();
    output_s_temp->set_meta_step_size(3e-3);
    output_s_temp->increment_reference();
    output_s_temp->set_utility_to_keep(utility_to_keep);
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
  if (targets.size() != this->output_neurons.size()) {
    std::cout << "Target size and the number of output neurons dont match\n";
    exit(1);
  }
  for (int counter = 0; counter < targets.size(); counter++) {
    error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step, gamma, lambda);
  }
  return error * error;
}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets,
                                                    float gamma,
                                                    float lambda,
                                                    std::vector<bool> no_grad) {
  float error = 0;
  if (targets.size() != 1) {
    std::cout << "Multiple target values passed. This network only learns to make a single prediction.\n";
    exit(1);
  }
  for (int counter = 0; counter < targets.size(); counter++) {
    error += this->output_neurons[counter]->introduce_targets(targets[counter],
                                                              this->time_step,
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

