//
// Created by Khurram Javed on 2021-09-22.
//

#ifndef INCLUDE_NN_NETWORKS_IMPRINTING_MNIST_H_
#define INCLUDE_NN_NETWORKS_IMPRINTING_MNIST_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synced_synapse.h"
#include "../synced_neuron.h"
#include "../dynamic_elem.h"
#include "./synced_network.h"

class ImprintingMNIST : public SyncedNetwork {

 public:

  std::vector<SyncedSynapse *> active_synapses;

  std::vector<LTUSynced *> LTU_neurons;

  ImprintingMNIST(float step_size, int seed, int no_of_input_features, float utility_to_keep, int hidden_units);

  ~ImprintingMNIST();

  void print_graph(SyncedNeuron *root);

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void imprint();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets);

  void add_feature(float step_size, float utility_to_keep);

  void add_feature_binary(float step_size, float utility_to_keep);

  void imprint_feature(int index, std::vector<float> feature);
};

#endif //INCLUDE_NN_NETWORKS_IMPRINTING_MNIST_H_
