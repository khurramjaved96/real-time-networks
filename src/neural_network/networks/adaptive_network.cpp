//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//


#include "../../../include/neural_networks/networks/adaptive_network.h"
#include <assert.h>
#include <cmath>
#include <random>
#include <execution>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
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

int ContinuallyAdaptingNetwork::get_total_neurons() {
    int tot = 0;
    for (auto it : this->all_neurons) {
        if (it->is_mature)
            tot++;
    }
    return tot;
}

ContinuallyAdaptingNetwork::ContinuallyAdaptingNetwork(float step_size, int seed, int no_of_input_features) : mt(seed) {
    this->time_step = 0;

//  Initialize the neural network input neurons.
//  Currently we fix an input size of 10.
    int input_neuron = no_of_input_features;

    for (int counter = 0; counter < input_neuron; counter++) {
        auto n = new neuron(false, false, true);
        n->is_mature = true;
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
        n->increment_reference();
        this->input_neurons.push_back(n);
        n->increment_reference();
        this->all_neurons.push_back(n);
    }

//  Initialize all output neurons.
//  Similarly, we fix an output size to 1.

    this->error_neuron = new neuron(false, true);
    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(this->error_neuron));
    error_neuron->increment_reference();
    this->all_neurons.push_back(this->error_neuron);

    int output_neuros = 1;
    for (int counter = 0; counter < output_neuros; counter++) {
        auto n = new neuron(false, true);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
        n->increment_reference();
        this->output_neurons.push_back(n);
        n->increment_reference();
        this->all_neurons.push_back(n);
    }


//  Connect our input and output neurons with synapses.
    for (auto &input : this->input_neurons) {
        for (auto &output : this->output_neurons) {
            synapse *s = new synapse(input, output, 0, step_size);
            this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
            s->increment_reference();
            this->all_synapses.push_back(s);
            s->increment_reference();
            this->output_synapses.push_back(s);
            s->turn_on_idbd();
        }
    }
}

void ContinuallyAdaptingNetwork::print_graph(neuron *root) {
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

void ContinuallyAdaptingNetwork::viz_graph() {
    NetworkVisualizer netviz = NetworkVisualizer(this->all_neurons);
    netviz.generate_dot(this->time_step);
    netviz.generate_dot_detailed(this->time_step);
}

std::string ContinuallyAdaptingNetwork::get_viz_graph() {
    NetworkVisualizer netviz = NetworkVisualizer(this->all_neurons);
    return netviz.get_graph(this->time_step);
//    netviz.generate_dot_detailed(this->time_step);
}


long long int ContinuallyAdaptingNetwork::get_timestep() {
    return this->time_step;
}

/**
 * Add a feature by adding a neuron to the neural network. This neuron is connected
 * to each (non-output) neuron w.p. perc ~ U(0, 1) and connected to each output neuron
 * with either a -1 and 1 weight.
 * @param step_size: step size of the input synapse added. Step size of the output synapse added starts as 0.
 */
void ContinuallyAdaptingNetwork::add_feature(float step_size) {
//  Limit our number of synapses to 1m
    if (this->all_synapses.size() < 1000000) {
//        std::normal_distribution<float> dist(0, 1);
        std::uniform_int_distribution<int> drinking_dist(100000, 300000);
        std::uniform_real_distribution<float> dist(-2, 2);
        std::uniform_real_distribution<float> dist_u(0, 1);
        std::uniform_real_distribution<float> dist_recurren(0, 0.99);

//      Create our new neuron
//        neuron *last_neuron = new neuron(true);
        neuron *recurrent_neuron = new neuron(true, false, false);
//        recurrent_neuron->drinking_age = drinking_dist(this->mt);
        recurrent_neuron->is_recurrent_neuron = true;
        recurrent_neuron->increment_reference();
        recurrent_neuron->increment_reference();
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(recurrent_neuron));
        this->all_neurons.push_back(recurrent_neuron);
        this->error_predicting_neurons.push_back(recurrent_neuron);

//      w.p. perc, attach a random neuron (that's not an output neuron) to this neuron
        float perc = dist_u(mt);
        for (auto &n : this->all_neurons) {
            if (!n->is_output_neuron && n->is_mature) {
                if (dist_u(mt) < perc) {
                    auto syn = new synapse(n, recurrent_neuron, 0.001 * dist(this->mt), step_size);
                    syn->block_gradients();
                    syn->increment_reference();
                    this->all_synapses.push_back(syn);

                    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
                }
            }
        }
//        if(recurrent_neuron->incoming_synapses.size()==0){
//            std::cout << "Creating new feature with no incoming neurons\n";
//            exit(1);
//        }
        auto syn_2 = new synapse(recurrent_neuron, recurrent_neuron, dist_recurren(this->mt), step_size);
        syn_2->block_gradients();
        syn_2->set_connected_to_recurrence(true);
        recurrent_neuron->recurrent_synapse = syn_2;
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn_2));
        syn_2->increment_reference();
        this->all_synapses.push_back(syn_2);
        syn_2->increment_reference();

//        std::cout << last_neuron->incoming_synapses.size() << std::endl;
//        exit(1);


//      Attach this neuron to all output neurons.
////      Set its weight to either 1 or -1

        synapse *output_s_temp;
        if (dist(this->mt) > 0) {
            output_s_temp = new synapse(recurrent_neuron, this->output_neurons[0], 1, 0);
        } else {
            output_s_temp = new synapse(recurrent_neuron, this->output_neurons[0], -1, 0);
        }
        output_s_temp->set_shadow_weight(true);
        output_s_temp->increment_reference();
        this->all_synapses.push_back(output_s_temp);
        output_s_temp->increment_reference();
        this->output_synapses.push_back(output_s_temp);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_s_temp));
    }
}


void ContinuallyAdaptingNetwork::set_print_bool() {
    std::cout
            << "From\tTo\tGrad_queue_size\tFrom activations_size\tTo activations_size\tError grad_queue From\tCredit\n";
    for (auto &s : this->all_synapses)
        s->print_status = false;
}


int ContinuallyAdaptingNetwork::get_input_size() {
    return this->input_neurons.size();
}

int ContinuallyAdaptingNetwork::get_total_synapses() {
    int tot = 0;
    for (auto it : this->all_synapses) {
        if (it->output_neuron->is_mature)
            tot++;
    }
    return tot;
}

ContinuallyAdaptingNetwork::~ContinuallyAdaptingNetwork() {
    for (auto &it : this->all_heap_elements)
        delete it;
}


void ContinuallyAdaptingNetwork::set_input_values(std::vector<float> const &input_values) {
//    assert(input_values.size() == this->input_neurons.size());
    for (int i = 0; i < input_values.size(); i++) {
        if (i < this->input_neurons.size()) {
            this->input_neurons[i]->value_before_firing = input_values[i];
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
void ContinuallyAdaptingNetwork::step() {
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
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->mark_useless_weights();
            });

//  Delete our is_useless weights and neurons
    std::for_each(
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->prune_useless_weights();
            });

//  For all synapses, if the synapse is is_useless set it has 0 references. We remove it.

    std::for_each(
            std::execution::par_unseq,
            this->all_synapses.begin(),
            this->all_synapses.end(),
            [&](synapse *s) {
                if (s->is_useless) {
                    s->decrement_reference();
                }
            });
    auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
    this->all_synapses.erase(it, this->all_synapses.end());

//  Similarly for all outgoing synapses and neurons.
    std::for_each(
            std::execution::par_unseq,
            this->output_synapses.begin(),
            this->output_synapses.end(),
            [&](synapse *s) {
                if (s->is_useless) {
                    s->decrement_reference();
                }
            });
    it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_s);
    this->output_synapses.erase(it, this->output_synapses.end());


    std::for_each(
            std::execution::par_unseq,
            this->all_neurons.begin(),
            this->all_neurons.end(),
            [&](neuron *s) {
                if (s->useless_neuron) {
                    s->decrement_reference();
                }
            });

    auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
    this->all_neurons.erase(it_n, this->all_neurons.end());
//    }


    this->time_step++;
}


/**
 * Find all synapses and neurons with 0 references to them and delete them.
 */
void ContinuallyAdaptingNetwork::collect_garbage() {
    for (int temp = 0; temp < this->all_heap_elements.size(); temp++) {
        if (all_heap_elements[temp]->references == 0) {
            delete all_heap_elements[temp];
            all_heap_elements[temp] = nullptr;
        }
    }

    auto it = std::remove_if(this->all_heap_elements.begin(), this->all_heap_elements.end(), is_null_ptr);
    this->all_heap_elements.erase(it, this->all_heap_elements.end());
}

std::vector<float> ContinuallyAdaptingNetwork::read_output_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->output_neurons.size());
    for (auto &output_neuro : this->output_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}

std::vector<float> ContinuallyAdaptingNetwork::read_all_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->all_neurons.size());
    for (auto &output_neuro : this->all_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}


float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1);
    }
    return error;
}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda) {
//  Put all targets into our neurons.
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        if (counter == 1) {
            std::cout << "More than one output neuron not supported currently\n";
            exit(1);
        }
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1, gamma, lambda);
    }
    this->error_neuron->introduce_targets(error, this->time_step - 1, gamma, lambda);
    return error * error;
}
