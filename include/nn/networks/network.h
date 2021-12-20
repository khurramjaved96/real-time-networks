//
// Created by Khurram Javed on 2021-07-21.
//

#ifndef INCLUDE_NN_NETWORKS_NETWORK_H
#define INCLUDE_NN_NETWORKS_NETWORK_H

#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"

class Network {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
  std::vector<Neuron *> output_neurons;
  std::vector<synapse *> all_synapses;
  std::vector<synapse *> output_synapses;
  std::vector<dynamic_elem *> all_heap_elements;
  std::vector<Neuron *> all_neurons;
  std::vector<Neuron *> input_neurons;

  void collect_garbage();

  Network();

  ~Network();

  int64_t get_timestep();

  void set_input_values(std::vector<float> const &input_values);

  std::vector<float> forward(std::vector<float> x);

  void backward(std::vector<float> targets, float gamma, float lambda);

  void update_weights();

  void prune_weights();

  void update_utility();

  void tick();

  void step();

  std::vector<float> read_output_values();

  std::vector<float> read_all_values();

  float introduce_targets(std::vector<float> targets);

  float introduce_targets(std::vector<float> targets, float gamma, float lambda);

  float introduce_targets(float targets, float gamma, float lambda, std::vector<bool> no_grad);

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

  void increment_references(Neuron * n, int n_references);
  void increment_references(synapse * s, int n_references);

//    virtual void add_feature() = 0;
};

#endif  // INCLUDE_NN_NETWORKS_NETWORK_H
