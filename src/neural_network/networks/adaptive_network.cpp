//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//

#include "../../../include/neural_networks/networks/adaptive_network.h"
#include "../../../include/neural_networks/neuron.h"
#include "../../../include/neural_networks/synapse.h"
#include "../../../include/utils.h"
#include <assert.h>
#include <random>
#include <execution>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>

ContinuallyAdaptingNetwork::ContinuallyAdaptingNetwork(float step_size, int width, int seed) : mt(seed) {
    this->time_step = 0;

    int input_neuron = 10;
    for (int counter = 0; counter < input_neuron; counter++) {
        auto n = new neuron(false, false, true);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
        n->references++;
        this->input_neurons.push_back(n);
        n->references++;
        this->all_neurons.push_back(n);
    }

    int output_neuros = 1;
    for (int counter = 0; counter < output_neuros; counter++) {
        auto n = new neuron(false, true);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
        n->references++;
        this->output_neurons.push_back(n);
        n->references++;
        this->all_neurons.push_back(n);
    }

    for (auto &input: this->input_neurons) {
        for (auto &output: this->output_neurons) {
            synapse *s = new synapse(input, output, 0, step_size);
            this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
            s->references++;
            this->all_synapses.push_back(s);
            s->references++;
            this->output_synapses.push_back(s);
            s->turn_on_idbd();
        }
    }

//
//    float top_range = sqrt(2.0 / float(width));
    std::normal_distribution<float> dist(0, 1);
//    std::uniform_int_distribution<int> sparse_generator = std::uniform_int_distribution<int>(0, 1000);

//    std::vector<neuron *> neurons_so_far;

//
//    for (auto &n : this->all_neurons) {
//        if (n->outgoing_synapses.size() > 0) {
//            int total_incoming = n->incoming_synapses.size();
//            double scale = sqrt(2.0 / float(total_incoming));
//            for (auto s : n->incoming_synapses) {
//                s->weight = dist(mt) * scale;
//            }
//        }
//    }


}

void ContinuallyAdaptingNetwork::print_graph(neuron *root) {

    for (auto &os: root->outgoing_synapses) {
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

void ContinuallyAdaptingNetwork::add_feature(float step_size) {

    if (this->all_synapses.size() < 1000000) {
        std::normal_distribution<float> dist(0, 1);
        std::uniform_real_distribution<float> dist_u(0, 1);

        neuron *last_neuron = new neuron(true);
        last_neuron->references++;
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(last_neuron));
        this->all_neurons.push_back(last_neuron);

        float perc = dist_u(mt);
        for (auto &n : this->all_neurons) {
            if (!n->is_output_neuron and n->mature) {
                if (dist_u(mt) < perc) {

                    auto syn = new synapse(n, last_neuron, 0.001 * dist(this->mt), step_size);
                    syn->enable_logging = false;
                    syn->block_gradients();
                    syn->references++;
                    this->all_synapses.push_back(syn);
                    this->all_heap_elements.push_back(static_cast<dynamic_elem *>(syn));
                }
            }
        }


        for (auto &output_n : this->output_neurons) {
            synapse *output_s_temp;
            if (dist(this->mt) > 0) {
                output_s_temp = new synapse(last_neuron, output_n, 1, 0);
            } else {
                output_s_temp = new synapse(last_neuron, output_n, -1, 0);
            }
            output_s_temp->references++;
            this->all_synapses.push_back(output_s_temp);
            output_s_temp->references++;
            this->output_synapses.push_back(output_s_temp);
            this->all_heap_elements.push_back(static_cast<dynamic_elem *>(output_s_temp));
        }
    }


//    if (last_neuron->outgoing_synapses.size() > 0) {
//        int total_incoming = last_neuron->incoming_synapses.size();
//        double scale = sqrt(2.0 / float(total_incoming));
//        for (auto s : last_neuron->incoming_synapses) {
//            s->weight = dist(mt) * scale;
//        }
//    }



}

std::vector<float> CustomNetwork::get_memory_weights() {
    std::vector<float> my_vec;
    for (auto &s : memory_feature_weights) {
        my_vec.push_back(s->weight);
    }
    return my_vec;
}

void CustomNetwork::set_print_bool() {
    std::cout
            << "From\tTo\tGrad_queue_size\tFrom activations_size\tTo activations_size\tError grad_queue From\tCredit\n";
    for (auto &s : this->all_synapses)
        s->print_status = false;
}


int CustomNetwork::get_input_size() {
    return this->input_neurons.size();
}

int CustomNetwork::
get_total_synapses() {
    return this->all_synapses.size();
}

CustomNetwork::~CustomNetwork() {
    for (auto &it : this->all_neurons)
        delete it;
    for (auto &it : this->all_synapses)
        delete it;
}


void CustomNetwork::set_input_values(std::vector<float> const &input_values) {
//    assert(input_values.size() == this->input_neurons.size());
    for (int i = 0; i < input_values.size(); i++) {
        if (i < this->input_neurons.size())
            this->input_neurons[i]->temp_value = input_values[i];
    }
}

bool to_delete_s(synapse *s) {
    return s->useless;
}

bool to_delete_n(neuron *s) {
    return s->useless_neuron;
}


void CustomNetwork::step() {


//    Making copies of features and storing them for a while (for structural credit-assignment problem
// Verify if correct
//    this->set_print_bool();
//    std::cout << "Copying memory\n";

//    std::cout << "Fire neuron\n";


    std::for_each(
            std::execution::par_unseq,
            memories.begin(),
            memories.end(),
            [&](no_grad_synapse *s) {
                s->copy_activation(this->time_step);
            });


    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->fire(this->time_step);
            });




//    std::cout << "Update value\n";
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->update_value();
            });

//    std::cout << "Forward gradients\n";
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->forward_gradients();
            });


//    std::cout << "Propogate error\n";
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->propogate_error();
            });

//    std::cout << "Updating weights\n";

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

//    std::cout << "Making useless weights\n";
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->mark_useless_weights();
            });

//    std::cout << "Pruning wieghts in queues\n";
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->prune_useless_weights();
            });

//    std::cout << "Remving activation\n";
    auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
    this->all_synapses.erase(it, this->all_synapses.end());
    it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_s);
    this->output_synapses.erase(it, this->output_synapses.end());


    auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
    this->all_neurons.erase(it_n, this->all_neurons.end());

    it_n = std::remove_if(this->new_features.begin(), this->new_features.end(), to_delete_n);
    this->new_features.erase(it_n, this->new_features.end());
    this->time_step++;


}


std::vector<float> CustomNetwork::read_output_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->output_neuros.size());
    for (auto &output_neuro : this->output_neuros) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}

std::vector<float> CustomNetwork::read_all_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->all_neurons.size());
    for (auto &output_neuro : this->all_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}


float CustomNetwork::introduce_targets(std::vector<float> targets) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neuros[counter]->introduce_targets(targets[counter], this->time_step - 1);
    }
    return error;
}

float CustomNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neuros[counter]->introduce_targets(targets[counter], this->time_step - 1, gamma, lambda);
    }
    return error;
}