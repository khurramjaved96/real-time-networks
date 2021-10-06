//
// Created by Khurram Javed on 2021-09-28.
//

#ifndef INCLUDE_NN_NETWORKS_LAYERWISE_FEEDWORWARD_H_
#define INCLUDE_NN_NETWORKS_LAYERWISE_FEEDWORWARD_H_




#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synced_synapse.h"
#include "../synced_neuron.h"
#include "../dynamic_elem.h"
#include "./synced_network.h"

class LayerwiseFeedforward : public SyncedNetwork {

 public:

  std::vector<SyncedSynapse *> active_synapses;

  std::vector<std::vector<SyncedNeuron *>> LTU_neuron_layers;

  LayerwiseFeedforward(float step_size, int seed, int no_of_input_features, int total_targets, float utility_to_keep);

  ~LayerwiseFeedforward();

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

  void imprint_feature_random();
};


#endif //INCLUDE_NN_NETWORKS_LAYERWISE_FEEDWORWARD_H_
