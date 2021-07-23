//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//
#include "../include/fixed_recurrent_network.h"
#include <assert.h>
#include <cmath>


#include <random>
#include <execution>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>

#include "../../include/nn/neuron.h"
#include "../../include/nn/synapse.h"
#include "../../include/nn/dynamic_elem.h"
#include "../../include/utils.h"
#include "../../include/nn/utils.h"

/**
 * Continually adapting neural network.
 * Essentially a neural network with the ability to add and remove neurons
 * based on a generate and test approach.
 * Check the corresponding header file for a description of the variables.
 *
 * As a quick note as to how this NN works - it essentially fires all neurons once
 * per step, unlike a usual NN that does a full forward pass per output needed.
 *
 * @param step_size: neural network step size.
 * @param width: [NOT CURRENTLY USED] neural network width
 * @param seed: random seed to initialize.
 */


ContinuallyAdaptingRecurrentNetworkTest::ContinuallyAdaptingRecurrentNetworkTest(float step_size, int seed,
                                                                                 int no_of_input_features) {
    this->mt.seed(seed);
    this->time_step = 0;
    int input_neuron = 1;


    auto n = new neuron(false, false, true);
    n->is_mature = true;
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
    n->increment_reference();
    this->input_neurons.push_back(n);
    n->increment_reference();
    this->all_neurons.push_back(n);

    auto n2 = new neuron(false, false, true);
    n2->is_mature = true;
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n2));
    n2->increment_reference();
    this->input_neurons.push_back(n2);
    n2->increment_reference();
    this->all_neurons.push_back(n2);

    auto recurrent_neuron = new neuron(true, false, false);
    recurrent_neuron->is_recurrent_neuron = true;
    recurrent_neuron->is_mature = true;
    recurrent_neuron->increment_reference();
    recurrent_neuron->increment_reference();
    this->all_neurons.push_back(recurrent_neuron);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(recurrent_neuron));

    auto syn = new synapse(n, recurrent_neuron, 0.9, step_size);
    auto syn_1 = new synapse(n2, recurrent_neuron, -0.8, step_size);
    auto syn_2 = new synapse(recurrent_neuron, recurrent_neuron, 0.6, step_size);
    syn->block_gradients();
    syn_1->block_gradients();
    syn_2->block_gradients();
    syn_2->set_connected_to_recurrence(true);

    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
    syn->increment_reference();
    this->all_synapses.push_back(syn);
    syn->increment_reference();

    recurrent_neuron->recurrent_synapse = syn_2;
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn_1));
    syn_1->increment_reference();
    this->all_synapses.push_back(syn_1);
    syn_1->increment_reference();

    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn_2));
    syn_2->increment_reference();
    this->all_synapses.push_back(syn_2);
    syn_2->increment_reference();

//  Initialize all output neurons.
//  Similarly, we fix an output size to 1.


    auto output_n = new neuron(false, true);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_n));
    output_n->increment_reference();
    this->output_neurons.push_back(output_n);
    output_n->increment_reference();
    this->all_neurons.push_back(output_n);

    auto *s = new synapse(recurrent_neuron, output_n, 0.7, step_size);
    s->turn_on_idbd();
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
    s->increment_reference();
    this->all_synapses.push_back(s);
    s->increment_reference();
    this->output_synapses.push_back(s);


}



/**
 * Add a feature by adding a neuron to the neural network. This neuron is connected
 * to each (non-output) neuron w.p. perc ~ U(0, 1) and connected to each output neuron
 * with either a -1 and 1 weight.
 * @param step_size: step size of the input synapse added. Step size of the output synapse added starts as 0.
 */
void ContinuallyAdaptingRecurrentNetworkTest::add_feature(float step_size) {
    return;
}


ContinuallyAdaptingRecurrentNetworkTest::~ContinuallyAdaptingRecurrentNetworkTest() {
}


/**
 * Step function after putting in the inputs to the neural network.
 * This function takes a step in the NN by firing all neurons.
 * Afterwards, it calculates gradients based on previous error and
 * propagates it back. Currently backprop is truncated at 1 step.
 * Finally, it updates its weights and prunes is_useless neurons and synapses.
 */
void ContinuallyAdaptingRecurrentNetworkTest::step() {
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->fire(this->time_step);
            });

//  Calculate and temporarily hold our next neuron values.
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->update_value();
            });

//  Contrary to the name, this function passes gradients BACK to the incoming synapses
//  of each neuron.
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->forward_gradients();
            });

//  Now we propagate our error backwards one step
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
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

    this->time_step++;
}