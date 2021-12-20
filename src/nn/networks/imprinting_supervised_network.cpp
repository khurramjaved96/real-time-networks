//
// Created by Khurram Javed on 2021-09-20.
//

#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/imprinting_supervised_network.h"
#include "../../../include/nn/synced_neuron.h"

ImprintingSupervised::ImprintingSupervised(float step_size,
                                           int seed,
                                           int no_of_input_features,
                                           float utility_to_keep, int hidden_units) {

  this->mt.seed(seed);
  int HIDDEN_NEURONS = hidden_units;
  for (int i = 0; i < no_of_input_features; i++) {
    SyncedNeuron *n = new LinearSyncedNeuron(true, false);
    this->input_neurons.push_back(n);
  }

  for (int b = 0; b < HIDDEN_NEURONS; b++) {
    SyncedNeuron *n = new LTUSynced(false, false, 9.5);
    this->LTU_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int outer = 0; outer < no_of_input_features; outer++) {
    for (int inner = 0; inner < HIDDEN_NEURONS; inner++) {
      SyncedSynapse *s = new SyncedSynapse(this->input_neurons[outer], this->LTU_neurons[inner], 0, 0);
      this->active_synapses.push_back(s);
    }
  }
  SyncedNeuron *output_neuron = new LinearSyncedNeuron(false, true);
  this->output_neurons.push_back(output_neuron);

  for (int inner = 0; inner < HIDDEN_NEURONS; inner++) {
    SyncedSynapse *s = new SyncedSynapse(this->LTU_neurons[inner], output_neuron, 0, step_size);
//    s->turn_on_idbd();
//    s->set_meta_step_size(step_size);
    this->output_synapses.push_back(s);
  }
}


ImprintingSupervised::~ImprintingSupervised() {

}

void ImprintingSupervised::forward(std::vector<float> inp) {

  this->set_input_values(inp);

  std::for_each(
      std::execution::par_unseq,
      this->input_neurons.begin(),
      this->input_neurons.end(),
      [&](SyncedNeuron *n) {
        n->fire(this->time_step);
      });

  std::for_each(
      std::execution::par_unseq,
      this->LTU_neurons.begin(),
      this->LTU_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_value(this->time_step);
      });

//  std::cout << "Value\tValBeforeFiring\n";
//  for(auto n : this->LTU_neurons){
//    std::cout << n->value << "\t" << n->value_before_firing << std::endl;
//  }
  std::for_each(
      std::execution::par_unseq,
      LTU_neurons.begin(),
      LTU_neurons.end(),
      [&](SyncedNeuron *n) {
        n->fire(this->time_step);
      });

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

  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_utility();
      });

  std::for_each(
      std::execution::par_unseq,
      this->LTU_neurons.begin(),
      this->LTU_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_utility();
      });

  this->time_step++;
}

void ImprintingSupervised::backward(std::vector<float> target) {
  this->introduce_targets(target);

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



  std::for_each(
      std::execution::par_unseq,
      output_synapses.begin(),
      output_synapses.end(),
      [&](SyncedSynapse *s) {
        s->update_utility();
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

void ImprintingSupervised::imprint_feature(int index, std::vector<float> feature, float target, float step_size_new_feature, float threshold, float probability, float step_size) {
  int counter = 0;
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  int total_available_for_replacement = 0;
  if(prob_sampler(this->mt) < 0.99) {
//    std::cout << "Imprinting feature\n";
    int min_index = -1;
    float min_util = 1000000;

    for (auto it: this->LTU_neurons) {

      if(it->neuron_age > 5000){
        std::cout << "Neuron utility = " << it->neuron_utility << std::endl;
        total_available_for_replacement++;
      }
      if (min_index == -1 && it->neuron_age > 5000) {
        min_index = counter;
        min_util = it->neuron_utility;
      } else {
        if (it->neuron_utility < min_util && it->neuron_age > 5000) {
          min_util = it->neuron_utility;
          min_index = counter;
        }
      }
      counter++;
    }
//
    if(min_index!= -1 and total_available_for_replacement > this->LTU_neurons.size()*0.5) {
      counter = 0;
      this->LTU_neurons[min_index]->neuron_age = 0;
      this->LTU_neurons[min_index]->outgoing_synapses[0]->weight = 0;
      float value = this->output_neurons[0]->value;
      float grad = this->output_neurons[0]->backward(value);
//      s = new SyncedSynapse(new_neuron, this->output_neurons[i], (1-value)*grad*0.2, step_size);
      this->LTU_neurons[min_index]->outgoing_synapses[0]->weight = (target-value)*grad*step_size_new_feature;
//      this->LTU_neurons[min_index]->outgoing_synapses[0]->weight = 1000;

//      this->LTU_neurons[min_index]->outgoing_synapses[0]->step_size = step_size;
      int total_ones=  0;
      for (auto it: this->LTU_neurons[min_index]->incoming_synapses) {
        if(prob_sampler(this->mt) < probability) {
          if (feature[counter] == 1) {
//          std::cout << "Positive weight\n";
            total_ones++;
            it->weight = 1;
          } else if (feature[counter] == 0) {
//          std::cout << "Negative weight\n";
            it->weight = -1;
          }
        }
        counter++;
      }
      LTUSynced* ltu_neuron;
      ltu_neuron = static_cast<LTUSynced*> (this->LTU_neurons[min_index]);
      ltu_neuron->set_threshold(threshold*total_ones);
//      std::cout << "Done imprinting\n\n\n";
    }

  }
}