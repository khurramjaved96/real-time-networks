//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//

#include "../include/fixed_feedforward_network.h"
#include <assert.h>
#include <random>
#include <execution>
#include <iostream>
#include "../../include/neural_networks/neuron.h"
#include "../../include/neural_networks/synapse.h"
#include "../../include/neural_networks/utils.h"

TestCase::TestCase(float step_size, int width, int seed) {
    this->time_step = 0;

    int input_neuron = 3;
    for (int counter = 0; counter < input_neuron; counter++) {
        auto n = new neuron(false, false, true);
        this->input_neurons.push_back(n);
        this->all_neurons.push_back(n);
    }


    bool relu = true;
    auto n = new neuron(relu);
    this->all_neurons.push_back(n);

    n = new neuron(relu);
    this->all_neurons.push_back(n);

    n = new neuron(relu);
    this->all_neurons.push_back(n);

    int output_neuros = 1;
    for (int counter = 0; counter < output_neuros; counter++) {
        auto n = new neuron(false, true);
        this->output_neurons.push_back(n);
        this->all_neurons.push_back(n);
    }

    this->all_synapses.push_back(new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[1 - 1], all_neurons[6 - 1], 0.5, step_size));
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
            [&](neuron *n) {
                n->fire(this->time_step);
            });

    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->update_value();
            });

    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->forward_gradients();
            });


    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
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
