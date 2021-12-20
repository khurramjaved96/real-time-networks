#include "../../../include/nn/networks/continuous_generation_network.h"
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


ContinuousGenerationNetwork::ContinuousGenerationNetwork(int no_of_input_features,
                                                         int no_of_output_neurons,
                                                         bool make_linear_connections,
                                                         float step_size,
                                                         float meta_step_size,
                                                         bool tidbd,
                                                         int seed,
                                                         bool use_imprinting,
                                                         int imprinting_max_incoming_connections,
                                                         int imprinting_max_num_microstimuli,
                                                         int imprinting_max_new_features_per_step,
                                                         bool imprinting_only_single_layer,
                                                         int linear_drinking_age,
                                                         float linear_synapse_local_utility_trace_decay,
                                                         float linear_utility_to_keep
                                                         std::pair<int, int> short_term_feature_recycling_age) {

  Neuron::gen = std::mt19937(seed);
  this->time_step = 0;
  this->mt.seed(seed);
  this->use_imprinting = use_imprinting;
  this->step_size = step_size;
  this->meta_step_size = meta_step_size;
  this->imprinting_max_incoming_connections = imprinting_max_incoming_connections;
  this->imprinting_max_num_microstimuli = imprinting_max_num_microstimuli;
  this->imprinting_max_new_features_per_step = imprinting_max_new_features_per_step;
  this->imprinting_only_single_layer = imprinting_only_single_layer;
  this->short_term_feature_recycling_age = short_term_feature_recycling_age;

  std::vector<int> inp(no_of_input_features);
  std::iota(inp.begin(), inp.end(), 0);
  this->input_indices = inp;

  std::uniform_real_distribution<float> dist(0, 1);

  for (int neuron_no = 0; neuron_no < no_of_input_features; neuron_no++) {
    auto n = new LinearNeuron(true, false);
    n->is_mature = true;
    n->drinking_age = linear_drinking_age;
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
    increment_references(n, 2);
  }

  for (int neuron_no = 0; neuron_no < no_of_output_neurons; neuron_no++) {
    auto n = new LinearNeuron(false, true);
    n->is_mature = true;
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
    increment_references(n, 2);
  }

  if (make_linear_connections){
    for (auto &inp_it : this->input_neurons){
      this->linear_features.push_back(inp_it);
      //increment_references(inp_it, 1);
      for (auto &out_it : this->output_neurons){
        synapse *s = new synapse(inp_it,out_it, 0, this->step_size * 0.001);
        s->synapse_local_utility_trace_decay = linear_synapse_local_utility_trace_decay;
        //TODO initializing optimistically here
        s->synapse_local_utility_trace = linear_utility_to_keep;
        s->set_utility_to_keep(linear_utility_to_keep);
        s->turn_on_idbd();
        s->set_meta_step_size(meta_step_size);
        s->block_gradients();
        this->all_synapses.push_back(s);
        this->output_synapses.push_back(s);
        increment_references(s, 2);
        inp_it->n_linear_synapses += 1;
      }
    }
  }

  // bias unit that is always 0, just to make the output receive inputs at start
  this->bias_unit = new BiasNeuron();
  this->bias_unit->is_mature = false;
  this->bias_unit->is_bias_unit = true;
  this->all_neurons.push_back(bias_unit);
  increment_references(this->bias_unit, 2);

  for (auto &output : this->output_neurons) {
    synapse *s = new synapse(bias_unit, output, 0.0, 0);
    s->set_utility_to_keep(linear_utility_to_keep);
    //s->disable_utility = true;
    s->block_gradients();
    this->all_synapses.push_back(s);
    this->output_synapses.push_back(s);
    increment_references(s, 2);
    s->set_meta_step_size(0);
  }

}



void ContinuousGenerationNetwork::step() {
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

  std::for_each(
      std::execution::par_unseq,
      linear_features.begin(),
      linear_features.end(),
      [&](Neuron *n) {
        n->mark_useless_linear_weights();
      });

//  Delete our is_useless weights and neurons
  std::for_each(
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->prune_useless_weights();
      });

//  For all synapses, if the synapse is is_useless set it has 0 references. We remove it.
  std::for_each(
      std::execution::par_unseq,
      this->all_synapses.begin(),
      this->all_synapses.end(),
      [&](synapse *s) {
        if (s->is_useless) {
          s->decrement_reference();
        }
      });
  auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
  this->all_synapses.erase(it, this->all_synapses.end());

// TODO why is this here???
//  Similarly for all outgoing synapses and neurons.
  std::for_each(
      std::execution::par_unseq,
      this->output_synapses.begin(),
      this->output_synapses.end(),
      [&](synapse *s) {
        if (s->is_useless) {
          s->decrement_reference();
        }
      });
  it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_s);
  this->output_synapses.erase(it, this->output_synapses.end());

// not deleteing the input neurons, just removing them from this list if
// they have no linear weights so that we dont need to iterate over them in the future
  auto it_ln = std::remove_if(this->linear_features.begin(), this->linear_features.end(), to_delete_linear_n);
  this->linear_features.erase(it_ln, this->linear_features.end());

  std::for_each(
      std::execution::par_unseq,
      this->all_neurons.begin(),
      this->all_neurons.end(),
      [&](Neuron *s) {
        if (s->useless_neuron) {
          s->decrement_reference();
        }
      });
  auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
  this->all_neurons.erase(it_n, this->all_neurons.end());

  std::for_each(
      std::execution::par_unseq,
      this->imprinted_features.begin(),
      this->imprinted_features.end(),
      [&](Neuron *s) {
        if (s->useless_neuron) {
          s->decrement_reference();
        }
      });
  it_n = std::remove_if(this->imprinted_features.begin(), this->imprinted_features.end(), to_delete_n);
  this->imprinted_features.erase(it_n, this->imprinted_features.end());

  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->memory_leak_patch();
      });

//  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->memory_leak_patch();
      });

  this->time_step++;
}


void ContinuousGenerationNetwork::imprint_on_interesting_neurons(std::vector<Neuron *> interesting_neurons) {
  // randomly pick some neurons from the provided "interesting_neurons" to imprint on
  int max_number_of_incoming_connections = 50;
  std::uniform_real_distribution<float> prob_max_selection(0, this->imprinting_max_prob);
  std::uniform_real_distribution<float> prob_selection(0, 1);
  std::uniform_int_distribution<> drinking_age_sampler(2500, 5000);
  float percentage_to_look_at = prob_max_selection(this->mt);
  int total_ones = 0;
  auto new_feature = new LTU(false, false, 100000);
  new_feature->drinking_age = drinking_age_sampler(this->mt);
  for (auto &it : interesting_neurons){
    if(prob_selection(this->mt) < percentage_to_look_at && total_ones <= max_number_of_incoming_connections) {
      auto s = new synapse(it, new_feature, 1, 0);
      //s->set_utility_to_keep(utility_to_keep);
      //s->synapse_local_utility_trace = utility_to_keep;
      s->synapse_local_utility_trace = s->get_utility_to_keep();
      this->all_synapses.push_back(s);
      increment_references(s, 1);
      s->set_meta_step_size(0);
      total_ones++;
    }
  }

  if (total_ones != 0){
    this->all_neurons.push_back(new_feature);
    this->imprinted_features.push_back(new_feature);
    increment_references(new_feature, 2);
    std::uniform_real_distribution<float> thres_sampler(0, total_ones);
    new_feature->activation_threshold = thres_sampler(this->mt);

    //float imprinting_weight = -1 * this->output_neurons[0]->error_gradient.back().error;
    //float imprinting_weight = 0.0001 * prob_selection(this->mt);
    float imprinting_weight = 0;
    auto s = new synapse(new_feature, this->output_neurons[0], imprinting_weight, this->step_size);
    //s->set_utility_to_keep(utility_to_keep);
    //s->synapse_local_utility_trace = utility_to_keep;
    s->synapse_local_utility_trace = s->get_utility_to_keep();
    this->all_synapses.push_back(s);
    this->output_synapses.push_back(s);
    increment_references(s, 2);
    s->set_meta_step_size(this->meta_step_size);
    s->turn_on_idbd();
    s->block_gradients();
    s->trace = 1;
    //std::cout << "Interesting: " << interesting_neurons.size() << " selected: " << total_ones << " thresh: " << new_feature->activation_threshold << " weight: " << imprinting_weight << " total current features: " << this->imprinted_features.size() << " total syn: " << this->all_synapses.size() << std::endl;
  }
  else
    delete new_feature;
}


void ContinuousGenerationNetwork::imprint_with_microstimuli() {
  // TODO handle multiple feature generation in cpp so that we can handle creation of microstimuli at
  // at random generated features.
  std::uniform_int_distribution<> num_incoming_conn_sampler(1, this->imprinting_max_incoming_connections);
  std::uniform_int_distribution<> num_microstimuli_sampler(0, this->imprinting_max_num_microstimuli);
  std::uniform_int_distribution<> num_new_features_sampler(1, this->imprinting_max_new_features_per_step);

  int num_incoming_connections = num_incoming_conn_sampler(this->mt);
  int num_microstimuli = num_microstimuli_sampler(this->mt);
  int num_new_features = num_new_features_sampler(this->mt);

  // make num_new_features number of new LTU features from randomly picked mature units.
  std::vector <Neuron *> generated_features;
  for (int i = 0; i < num_new_features; i++) {
    std::vector <Neuron *> incoming_features;
    std::uniform_int_distribution<> index_sampler(0, this->all_neurons.size() -1);

    int counter = 0; // so that we dont sample endlessly if conditions not met
    int total_ones = 0;
    // TODO optimize this
    // TODO may not work well if active features are rare. Have a condition to check for active features then.
    while (incoming_features.size() < num_incoming_connections && counter < this->all_neurons.size()){
      counter += 1;
      auto n = this->all_neurons[index_sampler(this->mt)];
      if (n->is_mature && !n->useless_neuron && !n->is_output_neuron && (!this->imprinting_only_single_layer || n->is_input_neuron)){
        incoming_features.push_back(n);
        if (n->old_value == 1)
          total_ones += 1;
      }
    }

    if (total_ones < 1){
      std::cout << "no active features selected. not generating." << std::endl;
      continue;
    }

    auto generated_feature = this->make_LTU_feature(incoming_features);
    if (generated_feature != nullptr){
      generated_features.push_back(generated_feature);
      this->short_term_memory_features.push_back(generated_feature);
    }
  }

  // pick num_microstimuli number of newly generated features randomly and make a new
  // microstimuli feature out of them
  if (generated_features.size() && num_microstimuli){
    std::uniform_int_distribution<> generated_feature_index_sampler(0, this->generated_features.size() -1);
    for (int i = 0; i < num_microstimuli; i++) {
      auto n = generated_features[generated_feature_index_sampler(this->mt)];
      auto generated_microstimuli = this->make_microstimuli_feature(n);
      if (generated_microstimuli != nullptr)
        this->short_term_memory_features.push_back(generated_microstimuli);
    }
  }
}


void ContinuousGenerationNetwork::set_input_values(std::vector<float> const &input_values) {
  if (input_values.size() != this->input_neurons.size()){
    std::cout << input_values.size() << " : " << this->input_neurons.size() << std::endl;
    std::cout << "err size in set_input_values()" << std::endl;
    exit(1);
  }

  std::for_each(
      std::execution::par_unseq,
      this->input_indices.begin(),
      this->input_indices.end(),
      [&](int i) {
        this->input_neurons[i]->old_old_value = this->input_neurons[i]->old_value;
        this->input_neurons[i]->old_value = this->input_neurons[i]->value;
        this->input_neurons[i]->old_value_without_activation = this->input_neurons[i]->value;
        this->input_neurons[i]->value = input_values[i];
        this->input_neurons[i]->value_without_activation = input_values[i];
      });
}
