//
// Created by Khurram Javed on 2021-09-22.
//

//
// Created by Khurram Javed on 2021-09-20.
//

#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/imprinting_mnist.h"

ImprintingMNIST::ImprintingMNIST(float step_size,
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
    LTUSynced *n = new LTUSynced(false, false, 9.5);
    this->LTU_neurons.push_back(n);
    n->neuron_age = 5000;
  }

  for (int outer = 0; outer < no_of_input_features; outer++) {
    for (int inner = 0; inner < HIDDEN_NEURONS; inner++) {
      SyncedSynapse *s = new SyncedSynapse(this->input_neurons[outer], this->LTU_neurons[inner], 0, 0);
      this->active_synapses.push_back(s);
    }
  }
  for(int output_neurons = 0; output_neurons < 10; output_neurons ++) {
    SyncedNeuron *output_neuron = new LinearSyncedNeuron(false, true);
    this->output_neurons.push_back(output_neuron);
  }

  for (int inner = 0; inner < HIDDEN_NEURONS; inner++) {
    for(int outer = 0; outer < 10; outer++) {
      SyncedSynapse *s = new SyncedSynapse(this->LTU_neurons[inner], this->output_neurons[outer], 0, 1e-4);
      s->turn_on_idbd();
      s->set_meta_step_size(step_size);
      this->output_synapses.push_back(s);
    }
  }

  for(int i =0; i< no_of_input_features; i++){
    for(int outer = 0; outer < 10; outer++) {
      SyncedSynapse *s = new SyncedSynapse(this->input_neurons[i], this->output_neurons[outer], 0, 1e-4);
      s->turn_on_idbd();
      s->set_meta_step_size(step_size);
      this->output_synapses.push_back(s);
    }
  }
}


ImprintingMNIST::~ImprintingMNIST() {

}

void ImprintingMNIST::forward(std::vector<float> inp) {

  this->set_input_values(inp);

  std::for_each(
      std::execution::par_unseq,
      this->LTU_neurons.begin(),
      this->LTU_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_value(this->time_step);
      });

  std::for_each(
      std::execution::par_unseq,
      LTU_neurons.begin(),
      LTU_neurons.end(),
      [&](LTUSynced *n) {
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

  this->time_step++;
}

void ImprintingMNIST::backward(std::vector<float> target) {
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

void ImprintingMNIST::imprint_feature(int index, std::vector<float> feature) {
  int counter = 0;
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  float percentage_to_look_at = prob_sampler(this->mt);
  if(prob_sampler(this->mt) < 0.99) {
//    std::cout << "Imprinting feature\n";
    int min_index = -1;
    float min_util = 1000000;

    for (auto it: this->LTU_neurons) {
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
    int total_ones = 0;
    if(min_index!= -1) {
      counter = 0;
      this->LTU_neurons[min_index]->neuron_age = 0;
      for(int i = 0; i< 10; i++) {
        this->LTU_neurons[min_index]->outgoing_synapses[i]->weight = 0;
        this->LTU_neurons[min_index]->outgoing_synapses[i]->step_size = 1e-2;
      }
      for (auto it: this->LTU_neurons[min_index]->incoming_synapses) {
        if (feature[counter] == 1) {
//        if(prob_sampler(this->mt) < 0.1){
          if(prob_sampler(this->mt) > percentage_to_look_at) {
            total_ones++;
            it->weight = 1;
          }
        }
        else {
          if(prob_sampler(this->mt) > percentage_to_look_at) {
            it->weight = -1;
          }
        }
        counter++;
      }
      std::uniform_real_distribution<float> thres_sampler(0, total_ones);
      this->LTU_neurons[min_index]->activation_threshold = thres_sampler(this->mt);
    }
//    LTUSynced *n = static_cast<LTUSynced *>(this->LTU_neurons[min_index]);
//    std::uniform_real_distribution<float> thres_sampler(0, total_ones);
//    n->activation_threshold = thres_sampler(this->mt);
  }
}