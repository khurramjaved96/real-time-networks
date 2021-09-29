//
// Created by Khurram Javed on 2021-09-28.
//

#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/layerwise_feedworward.h"

LayerwiseFeedforward::LayerwiseFeedforward(float step_size,
                                           int seed,
                                           int no_of_input_features,
                                           float utility_to_keep) {

  this->mt.seed(seed);
  for (int i = 0; i < no_of_input_features; i++) {
    SyncedNeuron *n = new LinearSyncedNeuron(true, false);
    n->set_layer_number(0);
    this->input_neurons.push_back(n);
  }

  for (
      int output_neurons = 0;
      output_neurons < 10; output_neurons++) {
    SyncedNeuron *output_neuron = new LinearSyncedNeuron(false, true);
    this->output_neurons.
        push_back(output_neuron);
  }

  for (
      int i = 0;
      i < no_of_input_features;
      i++) {
    for (
        int outer = 0;
        outer < 10; outer++) {
      SyncedSynapse *s = new SyncedSynapse(this->input_neurons[i], this->output_neurons[outer], 0, 1e-4);
      s->
          turn_on_idbd();
      s->
          set_meta_step_size(step_size);
      this->output_synapses.
          push_back(s);
    }
  }
}

LayerwiseFeedforward::~LayerwiseFeedforward() {

}

void LayerwiseFeedforward::forward(std::vector<float> inp) {

  this->set_input_values(inp);

  for (auto LTU_neuron_list: this->LTU_neuron_layers) {
    std::for_each(
        std::execution::par_unseq,
        LTU_neuron_list.begin(),
        LTU_neuron_list.end(),
        [&](SyncedNeuron *n) {
          n->update_value(this->time_step);
        });

    std::for_each(
        std::execution::par_unseq,
        LTU_neuron_list.begin(),
        LTU_neuron_list.end(),
        [&](LTUSynced *n) {
          n->fire(this->time_step);
        });

  }

  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_value(this->time_step);
      });

  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->fire(this->time_step);
      });

  this->time_step++;
}

void LayerwiseFeedforward::backward(std::vector<float> target) {
  this->introduce_targets(target, 0, 0);

  std::for_each(
      std::execution::par_unseq,
      output_neurons.begin(),
      output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->forward_gradients();
      });

//  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      output_synapses.begin(),
      output_synapses.end(),
      [&](SyncedSynapse *s) {
        s->assign_credit();
      });

//  Update our weights (based on either normal update or IDBD update
  std::for_each(
      std::execution::par_unseq,
      output_synapses.begin(),
      output_synapses.end(),
      [&](SyncedSynapse *s) {
        s->update_weight();
      });

}

void LayerwiseFeedforward::imprint_feature(int index, std::vector<float> feature) {
  int counter = 0;
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  float percentage_to_look_at = prob_sampler(this->mt);
  int max_layer = 0;
  LTUSynced *new_neuron = new LTUSynced(false, false, 0);
  int total_ones = 0;
  for (auto n : this->input_neurons) {
    if (n->value == 1) {
      if (prob_sampler(this->mt) < percentage_to_look_at) {
        this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, 1, 0));
        max_layer = max(max_layer, n->get_layer_number());
      }
    }
    else{
      if (prob_sampler(this->mt) < percentage_to_look_at*0.2) {
        this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, -1, 0));
        max_layer = max(max_layer, n->get_layer_number());
      }
    }
  }
  for (auto LTU_neuron_layer : this->LTU_neuron_layers) {
    for (auto n : LTU_neuron_layer) {
      if (n->value == 1 and n->neuron_age > 5000) {
        if (prob_sampler(this->mt) < percentage_to_look_at) {
          this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, 1, 0));
          max_layer = max(max_layer, n->get_layer_number());
        }
      }
      else if(n->neuron_age > 5000){
        if (prob_sampler(this->mt) < percentage_to_look_at*0.2) {
          this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, -1, 0));
          max_layer = max(max_layer, n->get_layer_number());
        }
      }
    }
  }
  if (new_neuron->incoming_synapses.size() > 0) {
    new_neuron->set_layer_number(max_layer + 1);
    for (int i = 0; i < this->output_neurons.size(); i++) {
      SyncedSynapse *s = new SyncedSynapse(new_neuron, this->output_neurons[i], 0, 1e-4);
      s->turn_on_idbd();
      s->set_meta_step_size(3e-2);
      this->output_synapses.push_back(s);
    }
//    LTUSynced *new_neuron_LTU = static_cast<LTUSynced *>(new_neuron);
    std::uniform_real_distribution<float> thres_sampler(0, new_neuron->incoming_synapses.size() - 0.5);
    new_neuron->activation_threshold = thres_sampler(this->mt);

    if (this->LTU_neuron_layers.size() >= (max_layer + 1)) {
      this->LTU_neuron_layers[max_layer + 1 - 1].push_back(new_neuron);
    } else {
      if (this->LTU_neuron_layers.size() != max_layer) {
        std::cout
            << "Should not happen; some neurons at layer k have not been added to appropritate vector of vector\n";
        exit(1);
      }
      std::vector < LTUSynced * > new_list;
      new_list.push_back(new_neuron);
      this->LTU_neuron_layers.push_back(new_list);
    }
  }
}