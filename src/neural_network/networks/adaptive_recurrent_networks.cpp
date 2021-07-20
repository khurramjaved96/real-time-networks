//
// Created by Khurram Javed on 2021-04-01.
//


#include <assert.h>
#include <cmath>

#include <random>
#include <execution>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>


#include "../../../include/neural_networks/networks/adaptive_recurrent_network.h"
#include "../../../include/neural_networks/neuron.h"
#include "../../../include/neural_networks/synapse.h"
#include "../../../include/neural_networks/dynamic_elem.h"
#include "../../../include/utils.h"
#include "../../../include/neural_networks/utils.h"

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

int ContinuallyAdaptingRecurrentNetwork::get_total_neurons() {
    return this->all_neurons.size();
}

ContinuallyAdaptingRecurrentNetwork::ContinuallyAdaptingRecurrentNetwork(float step_size, int seed,
                                                                         int no_of_input_features) : mt(seed) {
    this->time_step = 0;

//  Initialize the neural network input neurons.
//  Currently we fix an input size of 10.
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
    auto output_n = new neuron(false, true);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_n));
    output_n->increment_reference();
    this->output_neurons.push_back(output_n);
    output_n->increment_reference();
    this->all_neurons.push_back(output_n);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (int i = 0; i < 2; i++) {
        auto recurrent_neuron = new neuron(true, false, false);
        recurrent_neuron->is_recurrent_neuron = true;
        recurrent_neuron->is_mature = true;
        recurrent_neuron->increment_reference();
        recurrent_neuron->increment_reference();
        this->all_neurons.push_back(recurrent_neuron);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(recurrent_neuron));
        synapse *syn;
        synapse *syn_1;
        synapse *syn_2;
        if (i == 1) {
//            std::cout << "Gets in 1\n";
            syn = new synapse(n, recurrent_neuron, dist(this->mt), step_size);
            syn_1 = new synapse(n2, recurrent_neuron, dist(this->mt), step_size);
            syn_2 = new synapse(recurrent_neuron, recurrent_neuron, dist(this->mt), step_size);
        } else {
//            std::cout << "Gets in 2\n";
            syn = new synapse(n, recurrent_neuron, dist(this->mt), step_size);
            syn_1 = new synapse(n2, recurrent_neuron, dist(this->mt), step_size);
            syn_2 = new synapse(recurrent_neuron, recurrent_neuron, dist(this->mt), step_size);
        }
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


        synapse *s;
        if (i == 0)
            s = new synapse(recurrent_neuron, output_n, 0, step_size);
        else
            s = new synapse(recurrent_neuron, output_n, -0, step_size);
        s->turn_on_idbd();
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
        s->increment_reference();
        this->all_synapses.push_back(s);
        s->increment_reference();
        this->output_synapses.push_back(s);
    }
//    exit(1);
    synapse *s;

    s = new synapse(n, output_n, 0.3, step_size);
    s->turn_on_idbd();
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
    s->increment_reference();
    this->all_synapses.push_back(s);
    s->increment_reference();
    this->output_synapses.push_back(s);

    s = new synapse(n2, output_n, 0.3, step_size);
    s->turn_on_idbd();
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
    s->increment_reference();
    this->all_synapses.push_back(s);
    s->increment_reference();
    this->output_synapses.push_back(s);
}

void ContinuallyAdaptingRecurrentNetwork::print_graph(neuron *root) {
    for (auto &os : root->outgoing_synapses) {
        auto current_n = os;

        if (!current_n->print_status) {
            std::cout << current_n->input_neuron->id << "\t" << current_n->output_neuron->id << "\t"
                      << os->grad_queue.size() << "\t\t" << current_n->input_neuron->past_activations.size()
                      << "\t\t\t" << current_n->output_neuron->past_activations.size() << "\t\t\t"
                      << current_n->input_neuron->error_gradient.size()
                      << "\t\t" << current_n->credit << std::endl;
            current_n->print_status = true;
        }
        print_graph(current_n->output_neuron);
    }
}

void ContinuallyAdaptingRecurrentNetwork::viz_graph() {
    NetworkVisualizer netviz = NetworkVisualizer(this->all_neurons);
    netviz.generate_dot(this->time_step);
    netviz.generate_dot_detailed(this->time_step);
}

std::string ContinuallyAdaptingRecurrentNetwork::get_viz_graph() {
    NetworkVisualizer netviz = NetworkVisualizer(this->all_neurons);
    return netviz.get_graph(this->time_step);
//    netviz.generate_dot_detailed(this->time_step);
}


long long int ContinuallyAdaptingRecurrentNetwork::get_timestep() {
    return this->time_step;
}

/**
 * Add a feature by adding a neuron to the neural network. This neuron is connected
 * to each (non-output) neuron w.p. perc ~ U(0, 1) and connected to each output neuron
 * with either a -1 and 1 weight.
 * @param step_size: step size of the input synapse added. Step size of the output synapse added starts as 0.
 */
void ContinuallyAdaptingRecurrentNetwork::add_feature(float step_size) {
//  Limit our number of synapses to 1m
    if (this->all_synapses.size() < 1000000) {
//        std::normal_distribution<float> dist(0, 1);
        std::uniform_real_distribution<float> dist(-2, 2);
        std::uniform_real_distribution<float> dist_u(0, 1);

//      Create our new neuron
        neuron *last_neuron = new neuron(true);
        last_neuron->increment_reference();
        last_neuron->increment_reference();
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(last_neuron));
        this->all_neurons.push_back(last_neuron);
        this->error_predicting_neurons.push_back(last_neuron);

//      w.p. perc, attach a random neuron (that's not an output neuron) to this neuron
        float perc = dist_u(mt);
        for (auto &n : this->all_neurons) {
            if (!n->is_output_neuron && n->is_mature) {
                if (dist_u(mt) < perc) {
                    auto syn = new synapse(n, last_neuron, 0.001 * dist(this->mt), step_size);
//                    auto syn = new synapse(n, last_neuron, 0.001, step_size);
//                    auto syn = new synapse(n, last_neuron, 0.5 * dist(this->mt), step_size);
//
//                    auto syn = new synapse(n, last_neuron, 0.001, step_size);
                    syn->turn_on_idbd();
                    syn->block_gradients();
                    syn->increment_reference();
//                    syn->set_shadow_weight(true);
                    this->all_synapses.push_back(syn);
//                    std::cout << this->all_synapses.size() << std::endl;
//                    exit(1);
                    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
//                    exit(1);
                }
            }
        }
        auto syn = new synapse(last_neuron, last_neuron, 0.1, step_size);
        syn->block_gradients();
        syn->increment_reference();
        this->all_synapses.push_back(syn);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
//        std::cout << last_neuron->incoming_synapses.size() << std::endl;
//        exit(1);


//      Attach this neuron to all output neurons.
////      Set its weight to either 1 or -1

        synapse *output_s_temp;
        if (dist(this->mt) > 0) {
            output_s_temp = new synapse(last_neuron, this->output_neurons[0], 1, 0);
        } else {
            output_s_temp = new synapse(last_neuron, this->output_neurons[0], -1, 0);
        }
        output_s_temp->set_shadow_weight(true);
        output_s_temp->increment_reference();
        this->all_synapses.push_back(output_s_temp);
        output_s_temp->increment_reference();
        this->output_synapses.push_back(output_s_temp);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_s_temp));
    }
}


void ContinuallyAdaptingRecurrentNetwork::set_print_bool() {
    std::cout
            << "From\tTo\tGrad_queue_size\tFrom activations_size\tTo activations_size\tError grad_queue From\tCredit\n";
    for (auto &s : this->all_synapses)
        s->print_status = false;
}


int ContinuallyAdaptingRecurrentNetwork::get_input_size() {
    return this->input_neurons.size();
}

int ContinuallyAdaptingRecurrentNetwork::get_total_synapses() {
    return this->all_synapses.size();
}

ContinuallyAdaptingRecurrentNetwork::~ContinuallyAdaptingRecurrentNetwork() {
    for (auto &it : this->all_heap_elements)
        delete it;
}


void ContinuallyAdaptingRecurrentNetwork::set_input_values(std::vector<float> const &input_values) {
//    assert(input_values.size() == this->input_neurons.size());
    for (int i = 0; i < input_values.size(); i++) {
        if (i < this->input_neurons.size()) {
            this->input_neurons[i]->value_before_firing = input_values[i];
//            std::cout << "Setting gradient_activation = " << input_values[i] << std::endl;
        } else {
            std::cout << "More input features than input neurons\n";
            exit(1);
        }
    }
}


/**
 * Step function after putting in the inputs to the neural network.
 * This function takes a step in the NN by firing all neurons.
 * Afterwards, it calculates gradients based on previous error and
 * propagates it back. Currently backprop is truncated at 1 step.
 * Finally, it updates its weights and prunes is_useless neurons and synapses.
 */
void ContinuallyAdaptingRecurrentNetwork::step() {
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

//  Update our weights (based on either normal update or IDBD update
    std::for_each(
            std::execution::par_unseq,
            all_synapses.begin(),
            all_synapses.end(),
            [&](synapse *s) {
                s->update_weight();
            });

//  Mark all is_useless weights and neurons for deletion
//    std::for_each(
//            std::execution::par_unseq,
//            all_neurons.begin(),
//            all_neurons.end(),
//            [&](neuron *n) {
//                n->mark_useless_weights();
//            });
//
////  Delete our is_useless weights and neurons
//    std::for_each(
//            all_neurons.begin(),
//            all_neurons.end(),
//            [&](neuron *n) {
//                n->prune_useless_weights();
//            });

//  For all synapses, if the synapse is is_useless set it has 0 references. We remove it.

//    std::for_each(
//            std::execution::par_unseq,
//            this->all_synapses.begin(),
//            this->all_synapses.end(),
//            [&](synapse *s) {
//                if (s->is_useless) {
//                    s->decrement_reference();
//                }
//            });
//    auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
//    this->all_synapses.erase(it, this->all_synapses.end());
//
////  Similarly for all outgoing synapses and neurons.
//    std::for_each(
//            std::execution::par_unseq,
//            this->output_synapses.begin(),
//            this->output_synapses.end(),
//            [&](synapse *s) {
//                if (s->is_useless) {
//                    s->decrement_reference();
//                }
//            });
//    it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_s);
//    this->output_synapses.erase(it, this->output_synapses.end());
//
//
//    std::for_each(
//            std::execution::par_unseq,
//            this->all_neurons.begin(),
//            this->all_neurons.end(),
//            [&](neuron *s) {
//                if (s->useless_neuron) {
//                    s->decrement_reference();
//                }
//            });
//
//    auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
//    this->all_neurons.erase(it_n, this->all_neurons.end());
//    }


    this->time_step++;
}


/**
 * Find all synapses and neurons with 0 references to them and delete them.
 */
void ContinuallyAdaptingRecurrentNetwork::collect_garbage() {
    for (int temp = 0; temp < this->all_heap_elements.size(); temp++) {
        if (all_heap_elements[temp]->references == 0) {
            delete all_heap_elements[temp];
            all_heap_elements[temp] = nullptr;
        }
    }

    auto it = std::remove_if(this->all_heap_elements.begin(), this->all_heap_elements.end(), is_null_ptr);
    this->all_heap_elements.erase(it, this->all_heap_elements.end());
}

std::vector<float> ContinuallyAdaptingRecurrentNetwork::read_output_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->output_neurons.size());
    for (auto &output_neuro : this->output_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}

std::vector<float> ContinuallyAdaptingRecurrentNetwork::read_all_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->all_neurons.size());
    for (auto &output_neuro : this->all_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}


float ContinuallyAdaptingRecurrentNetwork::introduce_targets(std::vector<float> targets) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1);
    }
    return error;
}

float ContinuallyAdaptingRecurrentNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda) {
//  Put all targets into our neurons.
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        if (counter == 1) {
            std::cout << "More than one output neuron not supported currently\n";
            exit(1);
        }
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1, gamma, lambda);
    }
    return error * error;
}
