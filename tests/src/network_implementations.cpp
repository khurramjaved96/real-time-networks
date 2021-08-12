//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//

#include "../include/test_case_networks.h"
#include <assert.h>
#include <random>
#include <execution>
#include <iostream>
#include "../../include/nn/neuron.h"
#include "../../include/nn/synapse.h"
#include "../../include/nn/utils.h"

LambdaReturnNetwork::LambdaReturnNetwork() {

  this->time_step = 0;

  int input_neuron = 3;
  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  bool relu = true;
  auto n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  this->all_synapses.push_back(new synapse(all_neurons[1 - 1], all_neurons[4 - 1], -0.2, 0));
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], 0.6, 0));

  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, 0));

  this->all_synapses.push_back(new synapse(all_neurons[1 - 1], all_neurons[5 - 1], -0.42, 0));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.9, 0));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.7, 0));
  this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[6 - 1], 0.92, 0));

  for (auto it : this->all_synapses) {
    sum_of_gradients.push_back(0);
  }

}


IDBDLearningNetwork::IDBDLearningNetwork() {}

void IDBDLearningNetwork::step() {

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
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->forward_gradients();
      });

//  Now we propagate our error backwards one step
  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->propagate_error();
      });

//  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->assign_credit();
      });


  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->update_weight();
      });


  this->time_step++;


}
void LambdaReturnNetwork::step() {


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
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->forward_gradients();
      });

//  Now we propagate our error backwards one step
  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->propagate_error();
      });

//  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->assign_credit();
      });

  for (int counter = 0; counter < all_synapses.size(); counter++) {
    this->sum_of_gradients[counter] += all_synapses[counter]->credit;
  }

  this->time_step++;

}
TestCase::TestCase(float step_size, int width, int seed) {
  this->time_step = 0;

  int input_neuron = 3;
  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  bool relu = true;
  auto n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new ReluNeuron(false, false);
  this->all_neurons.push_back(n);

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  auto inp1 = new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size);
  auto inp2 = new synapse(all_neurons[1 - 1], all_neurons[6 - 1], 0.5, step_size);
  inp1->block_gradients();
  inp2->block_gradients();
  this->all_synapses.push_back(inp1);
  this->all_synapses.push_back(inp2);
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[5 - 1], 0.7, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[5 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[7 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[6 - 1], all_neurons[7 - 1], 0.2, step_size));

  for (auto it : this->all_synapses) {
    sum_of_gradients.push_back(0);
  }
}

TestCase::TestCase(float step_size, int width, int seed, bool sigmoid) {
  this->time_step = 0;

  int input_neuron = 3;
  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  bool relu = true;
  auto n = new SigmoidNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new SigmoidNeuron(false, false);
  this->all_neurons.push_back(n);

  n = new SigmoidNeuron(false, false);
  this->all_neurons.push_back(n);

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  auto inp1 = new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size);
  auto inp2 = new synapse(all_neurons[1 - 1], all_neurons[6 - 1], 0.5, step_size);
  inp1->block_gradients();
  inp2->block_gradients();
  this->all_synapses.push_back(inp1);
  this->all_synapses.push_back(inp2);
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[5 - 1], 0.7, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.1, step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[5 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[7 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[6 - 1], all_neurons[7 - 1], 0.2, step_size));

  for (auto it : this->all_synapses) {
    sum_of_gradients.push_back(0);
  }
}

TestCase::TestCase() {

}

MixedActivationTest::MixedActivationTest() {
  this->time_step = 0;
  float step_size = 0;
  int input_neuron = 3;
  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  bool relu = true;
  auto n = new LeakyRelu(false, false, 0.3);
  this->all_neurons.push_back(n);

  this->all_neurons.push_back(new LeakyRelu(false, false, 0.01));

  this->all_neurons.push_back(new ReluNeuron(false, false));

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  auto inp1 = new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size);
  auto inp2 = new synapse(all_neurons[1 - 1], all_neurons[6 - 1], 0.5, step_size);
  inp1->block_gradients();
  inp2->block_gradients();
  this->all_synapses.push_back(inp1);
  this->all_synapses.push_back(inp2);
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[5 - 1], 0.7, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[5 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[7 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[6 - 1], all_neurons[7 - 1], 0.2, step_size));

  for (auto it : this->all_synapses) {
    sum_of_gradients.push_back(0);
  }
}

LeakyReluTest::LeakyReluTest(float step_size, int width, int seed) {

  this->time_step = 0;

  int input_neuron = 3;
  for (int counter = 0; counter < input_neuron; counter++) {
    auto n = new LinearNeuron(true, false);
    this->input_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  bool relu = true;
  auto n = new LeakyRelu(false, false, 0.01);
  this->all_neurons.push_back(n);

  n = new LeakyRelu(false, false, 0.01);
  this->all_neurons.push_back(n);

  n = new LeakyRelu(false, false, 0.01);
  this->all_neurons.push_back(n);

  int output_neuros = 1;
  for (int counter = 0; counter < output_neuros; counter++) {
    auto n = new LinearNeuron(false, true);
    this->output_neurons.push_back(n);
    this->all_neurons.push_back(n);
  }

  auto inp1 = new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size);
  auto inp2 = new synapse(all_neurons[1 - 1], all_neurons[6 - 1], 0.5, step_size);
  inp1->block_gradients();
  inp2->block_gradients();
  this->all_synapses.push_back(inp1);
  this->all_synapses.push_back(inp2);
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[5 - 1], 0.7, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[7 - 1], -0.1, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[5 - 1], -0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[7 - 1], 0.2, step_size));
  this->all_synapses.push_back(new synapse(all_neurons[6 - 1], all_neurons[7 - 1], 0.2, step_size));

  for (auto it : this->all_synapses) {
    sum_of_gradients.push_back(0);
  }
}

void TestCase::step() {

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

  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->forward_gradients();
      });

  std::for_each(
      std::execution::par_unseq,
      all_neurons.begin(),
      all_neurons.end(),
      [&](Neuron *n) {
        n->propagate_deep_error();
      });

  //  Calculate our credit
  std::for_each(
      std::execution::par_unseq,
      all_synapses.begin(),
      all_synapses.end(),
      [&](synapse *s) {
        s->assign_credit();
      });

  for (int counter = 0; counter < all_synapses.size(); counter++) {
    this->sum_of_gradients[counter] += all_synapses[counter]->credit;
  }

  this->time_step++;
}

//ContinuallyAdaptingRecurrentNetworkTest::ContinuallyAdaptingRecurrentNetworkTest(float step_size, int seed,
//                                                                                 int no_of_input_features) {
//  this->mt.seed(seed);
//  this->time_step = 0;
//  int input_neuron = 1;
//
//  auto n = new LinearNeuron(true, false);
//  n->is_mature = true;
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
//  n->increment_reference();
//  this->input_neurons.push_back(n);
//  n->increment_reference();
//  this->all_neurons.push_back(n);
//
//  auto n2 = new LinearNeuron(true, false);
//  n2->is_mature = true;
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n2));
//  n2->increment_reference();
//  this->input_neurons.push_back(n2);
//  n2->increment_reference();
//  this->all_neurons.push_back(n2);
//
//  auto recurrent_neuron = new ReluNeuron(false, false);
//  recurrent_neuron->is_recurrent_neuron = true;
//  recurrent_neuron->is_mature = true;
//  recurrent_neuron->increment_reference();
//  recurrent_neuron->increment_reference();
//  this->all_neurons.push_back(recurrent_neuron);
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(recurrent_neuron));
//
//  auto syn = new synapse(n, recurrent_neuron, 0.9, step_size);
//  auto syn_1 = new synapse(n2, recurrent_neuron, -0.8, step_size);
//  auto syn_2 = new synapse(recurrent_neuron, recurrent_neuron, 0.6, step_size);
//  syn->block_gradients();
//  syn_1->block_gradients();
//  syn_2->block_gradients();
//  syn_2->set_connected_to_recurrence(true);
//
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
//  syn->increment_reference();
//  this->all_synapses.push_back(syn);
//  syn->increment_reference();
//
//  recurrent_neuron->recurrent_synapse = syn_2;
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn_1));
//  syn_1->increment_reference();
//  this->all_synapses.push_back(syn_1);
//  syn_1->increment_reference();
//
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn_2));
//  syn_2->increment_reference();
//  this->all_synapses.push_back(syn_2);
//  syn_2->increment_reference();
//
////  Initialize all output neurons.
////  Similarly, we fix an output size to 1.
//
//
//  auto output_n = new LinearNeuron(false, true);
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_n));
//  output_n->increment_reference();
//  this->output_neurons.push_back(output_n);
//  output_n->increment_reference();
//  this->all_neurons.push_back(output_n);
//
//  auto *s = new synapse(recurrent_neuron, output_n, 0.7, step_size);
//  s->turn_on_idbd();
//  this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
//  s->increment_reference();
//  this->all_synapses.push_back(s);
//  s->increment_reference();
//  this->output_synapses.push_back(s);
//
//}
//
//void ContinuallyAdaptingRecurrentNetworkTest::add_feature(float step_size) {
//  return;
//}
//
//ContinuallyAdaptingRecurrentNetworkTest::~ContinuallyAdaptingRecurrentNetworkTest() {
//}
//
//void ContinuallyAdaptingRecurrentNetworkTest::step() {
//  std::for_each(
//      std::execution::par_unseq,
//      all_neurons.begin(),
//      all_neurons.end(),
//      [&](Neuron *n) {
//        n->fire(this->time_step);
//      });
//
////  Calculate and temporarily hold our next neuron values.
//  std::for_each(
//      std::execution::par_unseq,
//      all_neurons.begin(),
//      all_neurons.end(),
//      [&](Neuron *n) {
//        n->update_value();
//      });
//
////  Contrary to the name, this function passes gradients BACK to the incoming synapses
////  of each neuron.
//  std::for_each(
//      std::execution::par_unseq,
//      all_neurons.begin(),
//      all_neurons.end(),
//      [&](Neuron *n) {
//        n->forward_gradients();
//      });
//
////  Now we propagate our error backwards one step
//  std::for_each(
//      std::execution::par_unseq,
//      all_neurons.begin(),
//      all_neurons.end(),
//      [&](Neuron *n) {
//        n->propagate_error();
//      });
//
////  Calculate our credit
//  std::for_each(
//      std::execution::par_unseq,
//      all_synapses.begin(),
//      all_synapses.end(),
//      [&](synapse *s) {
//        s->assign_credit();
//      });
//
//  this->time_step++;
//}
//




