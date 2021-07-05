//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//

#include "../../../include/neural_networks/networks/adaptive_network.h"
#include "../../../include/neural_networks/neuron.h"
#include "../../../include/neural_networks/synapse.h"
#include "../../../include/neural_networks/dynamic_elem.h"
#include "../../../include/utils.h"
#include <assert.h>
#include <random>
#include <execution>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>

ContinuallyAdaptingNetwork::ContinuallyAdaptingNetwork(float step_size, int num_input, int num_output, int seed) : mt(seed) {
    this->time_step = 0;

    for (int counter = 0; counter < num_input; counter++) {
        auto n = new neuron(false, false, true);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
        n->increment_reference();
        this->input_neurons.push_back(n);
        n->increment_reference();
        this->all_neurons.push_back(n);
    }

    for (int counter = 0; counter < num_output; counter++) {
        auto n = new neuron(false, true);
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(n));
        n->increment_reference();
        this->output_neurons.push_back(n);
        n->increment_reference();
        this->all_neurons.push_back(n);
    }

    for (auto &input: this->input_neurons) {
        for (auto &output: this->output_neurons) {
            synapse *s = new synapse(input, output, 0, step_size);
            this->all_heap_elements.push_back(static_cast<dynamic_elem *>(s));
            s->increment_reference();
            this->all_synapses.push_back(s);
            s->increment_reference();
            this->output_synapses.push_back(s);
            s->turn_on_idbd();
        }
    }

//
//    float top_range = sqrt(2.0 / float(width));
    std::normal_distribution<float> dist(0, 1);
//    std::uniform_int_distribution<int> sparse_generator = std::uniform_int_distribution<int>(0, 1000);

//    std::vector<neuron *> neurons_so_far;

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

    if (this->all_synapses.size() < 5000000) {
        std::normal_distribution<float> dist(0, 1);
        std::uniform_real_distribution<float> dist_u(0, 1);

        neuron *last_neuron = new neuron(true);
        last_neuron->increment_reference();
        this->all_heap_elements.push_back(static_cast<dynamic_elem *>(last_neuron));
        this->all_neurons.push_back(last_neuron);

        float perc = dist_u(mt);
        for (auto &n : this->all_neurons) {
            if (!n->is_output_neuron and n->mature) {
                if (dist_u(mt) < perc) {

                    auto syn = new synapse(n, last_neuron, 0.001 * dist(this->mt), step_size);
                    syn->enable_logging = false;
                    syn->block_gradients();
                    syn->increment_reference();
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
            output_s_temp->increment_reference();
            this->all_synapses.push_back(output_s_temp);
            output_s_temp->increment_reference();
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
    return this->all_synapses.size();
}

ContinuallyAdaptingNetwork::~ContinuallyAdaptingNetwork() {
    for (auto &it : this->all_heap_elements)
        delete it;
}


void ContinuallyAdaptingNetwork::set_input_values(std::vector<float> const &input_values) {
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

void ContinuallyAdaptingNetwork::reset_trace(){
    std::for_each(
            std::execution::par_unseq,
            all_synapses.begin(),
            all_synapses.end(),
            [&](synapse *s) {
                s->reset_trace();
            });
}

void ContinuallyAdaptingNetwork::step() {



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
                n->propogate_error();
            });


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


    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->mark_useless_weights();
            });

//    std::cout << "Pruning wieghts in queues\n";
    std::for_each(
//            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->prune_useless_weights();
            });


    std::for_each(
            std::execution::par_unseq,
            this->all_synapses.begin(),
            this->all_synapses.end(),
            [&](synapse *s) {
                if(s->useless) {
                    s->decrement_reference();
                }
            });
    auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
    this->all_synapses.erase(it, this->all_synapses.end());


    std::for_each(
            std::execution::par_unseq,
            this->output_synapses.begin(),
            this->output_synapses.end(),
            [&](synapse *s) {
                if(s->useless) {
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
                if(s->useless_neuron) {
                    s->decrement_reference();
                }
            });

    auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
    this->all_neurons.erase(it_n, this->all_neurons.end());


    this->time_step++;


}

bool is_null_ptr(dynamic_elem* elem)
{
//    return true;
    if(elem== nullptr)
        return true;
    return false;
}

void ContinuallyAdaptingNetwork::college_garbage() {

    for(int temp = 0; temp < this->all_heap_elements.size(); temp++){
//        if(all_heap_elements[temp]->references < 2) {
//            std::cout << all_heap_elements[temp]->references << std::endl;
//            exit(1);
//        }
        if(all_heap_elements[temp]->references == 0 ){
//            std::cout << "Deleting element\n";
            delete all_heap_elements[temp];
            all_heap_elements[temp] = nullptr;
//            exit(1);
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


//float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets) {
//    float error = 0;
//    for (int counter = 0; counter < targets.size(); counter++) {
//        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1);
//    }
//    return error;
//}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1, gamma, lambda);
    }
    return error;
}

float ContinuallyAdaptingNetwork::introduce_targets(std::vector<float> targets, float gamma, float lambda, std::vector<bool> no_grad) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step - 1, gamma, lambda, no_grad[counter]);
    }
    return error;
}
