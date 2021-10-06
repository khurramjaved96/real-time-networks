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
                                           int total_targets,
                                           float utility_to_keep) {

  this->mt.seed(seed);
  for (int i = 0; i < no_of_input_features; i++) {
    SyncedNeuron *n = new LinearSyncedNeuron(true, false);
    n->neuron_age = 10000000;
    n->set_layer_number(0);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  for (int output_neurons = 0; output_neurons < total_targets; output_neurons++) {
    SyncedNeuron *output_neuron = new SigmoidSyncedNeuron(false, true);
    output_neuron->set_layer_number(100);
    this->all_neurons.push_back(output_neuron);
    this->output_neurons.push_back(output_neuron);
  }

//  enable LFA
  for (int i = 0;i < no_of_input_features;i++) {
    for (int outer = 0; outer < total_targets; outer++) {
      SyncedSynapse *s = new SyncedSynapse(this->input_neurons[i], this->output_neurons[outer], 0, 1e-4);
      s->turn_on_idbd();
      s->set_meta_step_size(step_size);
      this->output_synapses.push_back(s);
      this->all_synapses.push_back(s);
    }
  }
  for(int i = 0; i<10; i++){
    std::vector<SyncedNeuron*> temp;
    this->LTU_neuron_layers.push_back(temp);
  }
}

LayerwiseFeedforward::~LayerwiseFeedforward() {

}

void LayerwiseFeedforward::forward(std::vector<float> inp) {

//  std::cout << "Set inputs\n";

  this->set_input_values(inp);

//  std::cout << "Firing\n";

    std::for_each(
        std::execution::par_unseq,
        this->input_neurons.begin(),
        this->input_neurons.end(),
        [&](SyncedNeuron *n) {
          n->fire(this->time_step);
        });



    int counter = 0;
  for (auto LTU_neuron_list: this->LTU_neuron_layers) {
    counter++;
//    std::cout << "Updating values " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        LTU_neuron_list.begin(),
        LTU_neuron_list.end(),
        [&](SyncedNeuron *n) {
          n->update_value(this->time_step);
        });

//    std::cout << "Firing " << counter << "\n";
    std::for_each(
        std::execution::par_unseq,
        LTU_neuron_list.begin(),
        LTU_neuron_list.end(),
        [&](SyncedNeuron *n) {
          n->fire(this->time_step);
        });

  }


//  std::cout << "Updating values output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_value(this->time_step);
      });

//  std::cout << "Firing output \n";
  std::for_each(
      std::execution::par_unseq,
      this->output_neurons.begin(),
      this->output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->fire(this->time_step);
      });

//  std::cout << "Updating neuron utility \n";
  std::for_each(
      std::execution::par_unseq,
      this->all_neurons.begin(),
      this->all_neurons.end(),
      [&](SyncedNeuron *n) {
        n->update_utility();
      });

  this->time_step++;
}

void LayerwiseFeedforward::backward(std::vector<float> target) {
  this->introduce_targets(target);

  std::for_each(
      std::execution::par_unseq,
      output_neurons.begin(),
      output_neurons.end(),
      [&](SyncedNeuron *n) {
        n->forward_gradients();
      });

  for (int layer = this->LTU_neuron_layers.size() - 1; layer >= 0; layer--) {
    std::for_each(
        std::execution::par_unseq,
        this->LTU_neuron_layers[layer].begin(),
        this->LTU_neuron_layers[layer].end(),
        [&](SyncedNeuron *n) {
          n->propagate_error();
        });

    std::for_each(
        std::execution::par_unseq,
        this->LTU_neuron_layers[layer].begin(),
        this->LTU_neuron_layers[layer].end(),
        [&](SyncedNeuron *n) {
          n->forward_gradients();
        });
  }
//  Calculate our credit

  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](SyncedSynapse *s) {
        s->update_utility();
      });


  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](SyncedSynapse *s) {
        s->assign_credit();
      });
//
////  Update our weights (based on either normal update or IDBD update
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](SyncedSynapse *s) {
        s->update_weight();
      });

//  if (this->time_step % 10 == 0) {
    std::for_each(
        this->all_neurons.begin(),
        this->all_neurons.end(),
        [&](SyncedNeuron *n) {
          n->mark_useless_weights();
        });

    std::for_each(
        this->all_neurons.begin(),
        this->all_neurons.end(),
        [&](SyncedNeuron *n) {
          n->prune_useless_weights();
        });

    int counter=0;
    for(int vector_ind = 0; vector_ind < this->LTU_neuron_layers.size(); vector_ind++){
//    for (auto LTU_neuron_list: this->LTU_neuron_layers) {
      counter++;
//      std::cout << "Mark: Layer " << counter << "\n";
//      std::for_each(
//          LTU_neuron_list.begin(),
//          LTU_neuron_list.end(),
//          [&](SyncedNeuron *n) {
//            if(n->useless_neuron){
////              std::cout << "USELESS NEURON LTU UNIT\n";
//            }
////            n->mark_useless_weights();
//          });
//
//
//
//
//    }
//    counter=0;
//    for (auto LTU_neuron_list: this->LTU_neuron_layers) {
//      counter++;
//      std::cout << "Prune: Layer " << counter << "\n";
//      std::for_each(
//          LTU_neuron_list.begin(),
//          LTU_neuron_list.end(),
//          [&](SyncedNeuron *n) {
//            n->prune_useless_weights();
//          });

      auto it_n = std::remove_if(this->LTU_neuron_layers[vector_ind].begin(), this->LTU_neuron_layers[vector_ind].end(), to_delete_synced_n);
      if(it_n != this->LTU_neuron_layers[vector_ind].end()){
//        std::cout << "Deleting unused neurons\n";
//        std::cout << this->LTU_neuron_layers[vector_ind].size() << std::endl;
        this->LTU_neuron_layers[vector_ind].erase(it_n, this->LTU_neuron_layers[vector_ind].end());
//        std::cout << this->LTU_neuron_layers[vector_ind].size() << std::endl;
      }
//      LTU_neuron_list.erase(it_n, LTU_neuron_list.end());
    }



    auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_synced_s);
    this->all_synapses.erase(it, this->all_synapses.end());

    it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_synced_s);
    this->output_synapses.erase(it, this->output_synapses.end());

    auto it_n_2 = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_synced_n);
    this->all_neurons.erase(it_n_2, this->all_neurons.end());
//    std::cout << "All neurons deleted\n";
//  }

}


void LayerwiseFeedforward::imprint_feature(int index, std::vector<float> feature) {
  int counter = 0;
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_real_distribution<float> prob_sampler_threshold(0, 0.1);
  std::uniform_int_distribution<int> prob_sampler_age(50, 5000);
  std::uniform_real_distribution<float> decay_rate(0.99, 0.99999);
  float decay_rate_val = decay_rate(this->mt);
  float percentage_to_look_at = prob_sampler_threshold(this->mt);
  int max_layer = 0;
  LTUSynced *new_neuron = new LTUSynced(false, false, 0);
//  new_neuron->drinking_age = prob_sampler_age(this->mt);
  new_neuron->drinking_age = prob_sampler_age(this->mt);
  this->all_neurons.push_back(static_cast<SyncedNeuron*>(new_neuron));
  int total_ones = 0;
  float util_value_temp = 0.0000001;
  for (auto n : this->input_neurons) {

    if (n->value == 1 && n->neuron_age > n->drinking_age && n->neuron_utility > 0) {
//      std::cout << "Gets here\n";
//      std::cout << n->value << "\n";
      if (prob_sampler(this->mt) < percentage_to_look_at) {
        total_ones++;
        auto new_synapse = new SyncedSynapse(n, new_neuron, 1, 0);
        new_synapse->trace_decay_rate = decay_rate_val;
        this->all_synapses.push_back(new_synapse);
        max_layer = max(max_layer, n->get_layer_number());
      }
    }
    else if( n->value == 0 && n->neuron_age > n->drinking_age  && n->neuron_utility > 0){
      if (prob_sampler(this->mt) < percentage_to_look_at) {
        auto new_synapse = new SyncedSynapse(n, new_neuron, -1, 0);
        new_synapse->trace_decay_rate = decay_rate_val;
        this->all_synapses.push_back(new_synapse);
        max_layer = max(max_layer, n->get_layer_number());
      }
    }
  }
  int depth= 0;
  for (auto LTU_neuron_layer : this->LTU_neuron_layers) {
    for (auto n : LTU_neuron_layer) {
      if (n->value == 1 && n->neuron_age > n->drinking_age && n->neuron_utility > 0) {
        if (prob_sampler(this->mt) < percentage_to_look_at) {
          total_ones++;
          auto new_synapse = new SyncedSynapse(n, new_neuron, 1, 0);
          new_synapse->trace_decay_rate = decay_rate_val;
          this->all_synapses.push_back(new_synapse);
          max_layer = max(max_layer, n->get_layer_number());
        }
      }
      else if(n->neuron_age > n->drinking_age &&  n->value == 0 && n->neuron_utility > 0){
        if (prob_sampler(this->mt) < percentage_to_look_at) {
          auto new_synapse = new SyncedSynapse(n, new_neuron, -1, 0);
          new_synapse->trace_decay_rate = decay_rate_val;
          this->all_synapses.push_back(new_synapse);
          max_layer = max(max_layer, n->get_layer_number());
        }
      }
    }
    depth+=1;
    if(depth == 10){
//      Maximum depth allowed : 6 layers
      break;
    }
  }
  if (total_ones > 0) {
    new_neuron->set_layer_number(max_layer + 1);
    for (int i = 0; i < this->output_neurons.size(); i++) {
      SyncedSynapse *s = new SyncedSynapse(new_neuron, this->output_neurons[i], 0, 1e-3);
      s->turn_on_idbd();
      s->trace_decay_rate = decay_rate_val;
      s->set_meta_step_size(3e-3);
      this->output_synapses.push_back(s);
      this->all_synapses.push_back(s);
    }
    LTUSynced *new_neuron_LTU = static_cast<LTUSynced *>(new_neuron);
    std::uniform_real_distribution<float> thres_sampler(max(0, total_ones - 10), total_ones - 0.5);
    new_neuron_LTU->activation_threshold = thres_sampler(this->mt);
//    std::cout << "Selected threshold " << new_neuron_LTU->activation_threshold << "\n";
//    exit(1);
    if (this->LTU_neuron_layers.size() >= (max_layer + 1)) {
      this->LTU_neuron_layers[max_layer + 1 - 1].push_back(new_neuron);
    } else {
      if (this->LTU_neuron_layers.size() != max_layer) {
        std::cout
            << "Should not happen; some neurons at layer k have not been added to appropritate vector of vector\n";
        exit(1);
      }
      std::vector < SyncedNeuron * > new_list;
      new_list.push_back(new_neuron);
      this->LTU_neuron_layers.push_back(new_list);
    }
  }
}

void LayerwiseFeedforward::imprint_feature_random() {
  int counter = 0;
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_real_distribution<float> prob_sampler_threshold(0, 0.1);
  std::uniform_real_distribution<float> sign_sampler(0, 1);
  std::uniform_int_distribution<int> prob_sampler_age(1000, 30000);
  std::uniform_real_distribution<float> decay_rate(0.99, 0.99999);
  float percentage_to_look_at = prob_sampler_threshold(this->mt);
  int max_layer = 0;
  float decay_rate_val = decay_rate(this->mt);
  LTUSynced *new_neuron = new LTUSynced(false, false, 0);
//  new_neuron->drinking_age = prob_sampler_age(this->mt);
  new_neuron->drinking_age = 4000;
  this->all_neurons.push_back(static_cast<SyncedNeuron*>(new_neuron));
  int total_ones = 0;
  float util_value_temp = 0.0000001;
  for (auto n : this->input_neurons) {

    if (sign_sampler(this->mt) > 0.5 && n->neuron_age > n->drinking_age && n->neuron_utility > n->incoming_synapses.size()*util_value_temp) {
//      std::cout << "Gets here\n";
//      std::cout << n->value << "\n";
      if (prob_sampler(this->mt) < percentage_to_look_at) {
        total_ones++;
        auto new_synapse = new SyncedSynapse(n, new_neuron, 1, 0);
        new_synapse->trace_decay_rate = decay_rate_val;
        this->all_synapses.push_back(new_synapse);
        max_layer = max(max_layer, n->get_layer_number());
      }
    }
    else if(  n->neuron_age > n->drinking_age  && n->neuron_utility > n->incoming_synapses.size()*util_value_temp){
      if (prob_sampler(this->mt) < percentage_to_look_at) {
        auto new_synapse = new SyncedSynapse(n, new_neuron, -1, 0);
        new_synapse->trace_decay_rate = decay_rate_val;
        this->all_synapses.push_back(new_synapse);
        max_layer = max(max_layer, n->get_layer_number());
      }
    }
  }
  int depth= 0;
  for (auto LTU_neuron_layer : this->LTU_neuron_layers) {
    for (auto n : LTU_neuron_layer) {
      if (sign_sampler(this->mt) > 0.5 && n->neuron_age > n->drinking_age && n->neuron_utility > n->incoming_synapses.size()*util_value_temp) {
        if (prob_sampler(this->mt) < percentage_to_look_at) {
          total_ones++;
          auto new_synapse = new SyncedSynapse(n, new_neuron, 1, 0);
          new_synapse->trace_decay_rate = decay_rate_val;
          this->all_synapses.push_back(new_synapse);
          max_layer = max(max_layer, n->get_layer_number());
        }
      }
      else if(n->neuron_age > n->drinking_age && n->neuron_utility > n->incoming_synapses.size()*util_value_temp){
        if (prob_sampler(this->mt) < percentage_to_look_at) {
          auto new_synapse = new SyncedSynapse(n, new_neuron, -1, 0);
          new_synapse->trace_decay_rate = decay_rate_val;
          this->all_synapses.push_back(new_synapse);
          max_layer = max(max_layer, n->get_layer_number());
        }
      }
    }
    depth+=1;
    if(depth == 9){
//      Maximum depth allowed : 6 layers
      break;
    }
  }
  if (total_ones > 0) {
    new_neuron->set_layer_number(max_layer + 1);
    for (int i = 0; i < this->output_neurons.size(); i++) {

      SyncedSynapse *s = new SyncedSynapse(new_neuron, this->output_neurons[i], 0, 1e-3);
      s->trace_decay_rate = decay_rate_val;
      s->turn_on_idbd();
      s->set_meta_step_size(3e-3);
      this->output_synapses.push_back(s);
      this->all_synapses.push_back(s);
    }
    LTUSynced *new_neuron_LTU = static_cast<LTUSynced *>(new_neuron);
    std::uniform_real_distribution<float> thres_sampler(max(0, total_ones - 10), total_ones - 0.5);
    new_neuron_LTU->activation_threshold = thres_sampler(this->mt);
//    std::cout << "Selected threshold " << new_neuron_LTU->activation_threshold << "\n";
//    exit(1);
    if (this->LTU_neuron_layers.size() >= (max_layer + 1)) {
      this->LTU_neuron_layers[max_layer + 1 - 1].push_back(new_neuron);
    } else {
      if (this->LTU_neuron_layers.size() != max_layer) {
        std::cout
            << "Should not happen; some neurons at layer k have not been added to appropritate vector of vector\n";
        exit(1);
      }
      std::vector < SyncedNeuron * > new_list;
      new_list.push_back(new_neuron);
      this->LTU_neuron_layers.push_back(new_list);
    }
  }
}



//void LayerwiseFeedforward::imprint_feature(int index, std::vector<float> feature) {
//  int counter = 0;
//  std::uniform_real_distribution<float> prob_sampler(0, 1);
//  std::uniform_int_distribution<int> prob_sampler_age(1000, 30000);
//  float percentage_to_look_at = prob_sampler(this->mt);
//  int max_layer = 0;
//  SyncedNeuron *new_neuron = new ReluSyncedNeuron(false, false);
////  new_neuron->drinking_age = prob_sampler_age(this->mt);
//  new_neuron->drinking_age = 4000;
//  this->all_neurons.push_back(new_neuron);
//  int total_ones = 0;
//  for (auto n : this->input_neurons) {
//    if (n->value == 1 && n->neuron_utility > 0.01 && n->neuron_age > n->drinking_age) {
//      if (prob_sampler(this->mt) < percentage_to_look_at) {
//        this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, 0.01, 1e-4));
//        max_layer = max(max_layer, n->get_layer_number());
//      }
//    } else {
//      if (prob_sampler(this->mt) < percentage_to_look_at * 0.2) {
//        this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, -0.01, 1e-4));
//        max_layer = max(max_layer, n->get_layer_number());
//      }
//    }
//  }
//  int depth = 0;
////  for (auto LTU_neuron_layer : this->LTU_neuron_layers) {
////    for (auto n : LTU_neuron_layer) {
////      if (n->value == 1 and n->neuron_age > 5000) {
////        if (prob_sampler(this->mt) < percentage_to_look_at) {
////          this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, 1, 0));
////          max_layer = max(max_layer, n->get_layer_number());
////        }
////      }
////      else if(n->neuron_age > 5000){
////        if (prob_sampler(this->mt) < percentage_to_look_at*0.2) {
////          this->all_synapses.push_back(new SyncedSynapse(n, new_neuron, -1, 0));
////          max_layer = max(max_layer, n->get_layer_number());
////        }
////      }
////    }
////    depth+=1;
////    if(depth == 3){
//////      Maximum depth allowed : 6 layers
////      break;
////    }
////  }
//  if (new_neuron->incoming_synapses.size() > 0) {
//    new_neuron->set_layer_number(max_layer + 1);
//    for (int i = 0; i < this->output_neurons.size(); i++) {
//      SyncedSynapse *s = new SyncedSynapse(new_neuron, this->output_neurons[i], 0, 1e-3);
//      s->turn_on_idbd();
//      s->set_meta_step_size(3e-3);
//      this->output_synapses.push_back(s);
//      this->all_synapses.push_back(s);
//    }
////    LTUSynced *new_neuron_LTU = static_cast<LTUSynced *>(new_neuron);
//    std::uniform_real_distribution<float> thres_sampler(0, new_neuron->incoming_synapses.size() - 0.5);
////    new_neuron->activation_threshold = thres_sampler(this->mt);
//
//    if (this->LTU_neuron_layers.size() >= (max_layer + 1)) {
//      this->LTU_neuron_layers[max_layer + 1 - 1].push_back(new_neuron);
//    } else {
//      if (this->LTU_neuron_layers.size() != max_layer) {
//        std::cout
//            << "Should not happen; some neurons at layer k have not been added to appropritate vector of vector\n";
//        exit(1);
//      }
//      std::vector < SyncedNeuron * > new_list;
//      new_list.push_back(new_neuron);
//      this->LTU_neuron_layers.push_back(new_list);
//    }
//  }
//}