//
// Created by Khurram Javed on 2021-09-20.
//

#ifndef INCLUDE_NN_NETWORKS_NETWORKSYNCED_H_
#define INCLUDE_NN_NETWORKS_NETWORKSYNCED_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synced_synapse.h"
#include "../synced_neuron.h"
#include "../dynamic_elem.h"

class SyncedNetwork {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
  std::vector<SyncedNeuron *> all_neurons;
  std::vector<SyncedNeuron *> output_neurons;
  std::vector<SyncedNeuron *> input_neurons;
  std::vector<SyncedSynapse *> all_synapses;
  std::vector<SyncedSynapse *> output_synapses;
  std::vector<dynamic_elem *> all_heap_elements;

  void collect_garbage();

  SyncedNetwork();

  ~SyncedNetwork();

  int64_t get_timestep();

  void set_input_values(std::vector<float> const &input_values);

  void step();

  std::vector<float> read_output_values();

  std::vector<float> read_all_values();

  float introduce_targets(std::vector<float> targets);

//  float introduce_targets(std::vector<float> targets, float gamma, float lambda);

//  float introduce_targets(float targets, float gamma, float lambda, std::vector<bool> no_grad);

  std::vector<float> forward_pass_without_side_effects(std::vector<float> input_vector);

  int get_input_size();

  void print_synapse_status();

  void print_neuron_status();

  int get_total_synapses();

  int get_total_neurons();

  void reset_trace();

  void print_graph(Neuron *root);

  void viz_graph();

  std::string get_viz_graph();

//    virtual void add_feature() = 0;
};

#endif //INCLUDE_NN_NETWORKS_NETWORKSYNCED_H_
