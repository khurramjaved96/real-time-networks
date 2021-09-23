//
// Created by taodav on 17/9/21.
//

#include "../../../include/nn/networks/adaptive_value_network.h"

AdaptiveValueNetwork::AdaptiveValueNetwork(int no_of_input_features, int no_output_neurons,
                                           int seed, float step_size,
                                           float meta_step_size, bool tidbd,
                                           float utility_to_keep = 0.001) {

  this->time_step = 0;
  this->mt.seed(seed);

  this->bias_unit = new BiasNeuron();
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

  int output_neuros = no_output_neurons;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    n->is_mature = true;
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
    n->increment_reference();
    this->output_neurons.push_back(n);
    n->increment_reference();
    this->all_neurons.push_back(n);
  }


//  Connect our input and output neurons with synapses.
  for (auto &input : this->input_neurons) {
    for (auto &output : this->output_neurons) {
      auto *s = new synapse(input, output, 0, step_size);

      this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
      s->increment_reference();
      this->all_synapses.push_back(s);
      s->increment_reference();
      this->output_synapses.push_back(s);
      s->set_utility_to_keep(utility_to_keep);
      s->set_meta_step_size(meta_step_size);
      if (tidbd) {
        s->turn_on_idbd();
      }
    }
  }
}

/**
 * Add a feature by adding a neuron to the neural network. This neuron is connected
 * to each (non-output) neuron w.p. perc ~ U(0, 1) and connected to each output neuron
 * with either a -1 and 1 weight.
 * @param step_size: step size of the input synapse added. Step size of the output synapse added starts as 0.
 */
void AdaptiveValueNetwork::add_feature(float step_size, float utility_to_keep = 0.001) {
//  Limit our number of synapses to 1m
  if (this->all_synapses.size() < 1000000) {
//        std::normal_distribution<float> dist(0, 1);
    std::uniform_int_distribution<int> drinking_dist(1000, 80000);
    std::uniform_int_distribution<int> random_int(0, this->all_neurons.size());
    std::uniform_int_distribution<int> max_features(1, 500);
    std::uniform_real_distribution<float> dist(-2, 2);
    std::uniform_real_distribution<float> dist_u(0, 1);
//    std::uniform_int_distribution<int> dist_u(0, 1);
    std::uniform_real_distribution<float> dist_recurren(0, 0.99);

//      Create our new neuron
    Neuron *new_feature = new ReluNeuron(false, false);
    new_feature->drinking_age = drinking_dist(this->mt) ;
    new_feature->increment_reference();
    new_feature->increment_reference();


    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(new_feature));
    this->all_neurons.push_back(new_feature);

    auto bias_syanpse = new synapse(this->bias_unit, new_feature,  -2.5, 0);
    bias_syanpse->turn_on_idbd();
    bias_syanpse->set_meta_step_size(0);
    bias_syanpse->block_gradients();
    bias_syanpse->increment_reference();
    bias_syanpse->set_utility_to_keep(utility_to_keep);
    this->all_synapses.push_back(bias_syanpse);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(bias_syanpse));

    float try_counter = 0;
    float total_features_selected = 0;
    int features_to_add = max_features(mt);
    features_to_add = 3;
    std::vector<int> pos_added;
    while(true){
      try_counter ++;
      if(try_counter == 1000 or total_features_selected == features_to_add)
        break;
      int pos = random_int(mt);
//      std::cout << "Pos = " << pos << std::endl;
      if(this->all_neurons[pos]->is_mature && !this->all_neurons[pos]->is_output_neuron){
        if(std::count(pos_added.begin(), pos_added.end(), pos) == 0) {
          pos_added.push_back(pos);
          auto syn = new synapse(this->all_neurons[pos], new_feature,   1.0, 0);
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
      output_s_temp = new synapse(new_feature, this->output_neurons[0], 0, step_size);
    } else {
      output_s_temp = new synapse(new_feature, this->output_neurons[0], 0, step_size);
    }
//    output_s_temp->set_shadow_weight(true);
//    output_s_temp->turn_off_idbd();
    output_s_temp->turn_on_idbd();
    output_s_temp->set_meta_step_size(1e-2);
    output_s_temp->increment_reference();
    output_s_temp->set_utility_to_keep(utility_to_keep);
    this->all_synapses.push_back(output_s_temp);
    output_s_temp->increment_reference();
    this->output_synapses.push_back(output_s_temp);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_s_temp));
  }
}
