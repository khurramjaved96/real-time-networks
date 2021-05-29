//
// Created by haseebs on 4/29/21.
//


#include "../../../include/neural_networks/networks/simple_network.h"
#include "../../../include/neural_networks/neuron.h"
#include "../../../include/neural_networks/synapse.h"
#include "../../../include/neural_networks/utils.h"
#include <assert.h>
#include <random>
#include <execution>
#include <iostream>

SimpleNetwork::SimpleNetwork(float step_size, int width, int seed) {
    this->time_step = 0;
    int n_input_neurons = 2;
    int n_output_neurons = 1;

    for (int counter = 0; counter < n_input_neurons; counter++) {
        auto n = new neuron(true);
        this->input_neurons.push_back(n);
        this->all_neurons.push_back(n);
    }

    std::vector<neuron*> neurons_so_far;
    for(int layer=0; layer < 100; layer++)
    {
        std::vector<neuron*> this_layer;
        for(int this_layer_neuron = 0; this_layer_neuron < width; this_layer_neuron++) {
            auto* n = new neuron(true);
            this->all_neurons.push_back(n);
            this_layer.push_back(n);
            //std::cout << layer << ":" << n->id << std::endl;
//            adding connections from input
            if(layer==0)
            {
                for (auto &it : this->input_neurons) {
                    this->all_synapses.push_back(new synapse(it, n, 0, step_size));
                }
            }
            for (auto &it : neurons_so_far) {
                this->all_synapses.push_back(new synapse(it, n, 0, step_size));
            }
        }
        while (!neurons_so_far.empty())
        {
            neurons_so_far.pop_back();
        }

        for(auto &it : this_layer){
            neurons_so_far.push_back(it);
        }
    }

    for (int counter=0; counter < n_output_neurons; counter++)
    {
        auto n = new neuron(false);
        this->output_neurons.push_back(n);
        this->all_neurons.push_back(n);
        for(auto &it : neurons_so_far){
            this->all_synapses.push_back(new synapse(it, n,  0, step_size));
        }
    }
}


void SimpleNetwork::print_graph(neuron *root) {

    for (auto &os: root->outgoing_synapses) {
        auto current_n = os;

        if (!current_n->print_status) {

            std::cout << current_n->input_neuron->id << "\t" << current_n->output_neuron->id << "\t"
                      << os->grad_queue.size() << "\t\t" << current_n->input_neuron->past_activations.size()
                      << "\t\t\t" << current_n->output_neuron->past_activations.size() << "\t\t\t"
                      << current_n->input_neuron->error_gradient.size()
                      << "\t\tc:" << current_n->credit << "\t\t\tw:" << os->weight << std::endl;
            current_n->print_status = true;
        }
        print_graph(current_n->output_neuron);
    }
}


void SimpleNetwork::set_print_bool() {
    std::cout
            << "From\tTo\tGrad_queue_size\tFrom activations_size\tTo activations_size\tError grad_queue From\tCredit\n";
    for (auto &s : this->all_synapses)
        s->print_status = false;
}


int SimpleNetwork::get_input_size() {
    return this->input_neurons.size();
}


int SimpleNetwork::
get_total_synapses() {
    return this->all_synapses.size();
}


SimpleNetwork::~SimpleNetwork() {
    for (auto &it : this->all_neurons)
        delete it;
    for (auto &it : this->all_synapses)
        delete it;
}


void SimpleNetwork::set_input_values(std::vector<float> const &input_values) {
    assert(input_values.size() == this->input_neurons.size());
    for (int i = 0; i < input_values.size(); i++) {
        this->input_neurons[i]->temp_value = input_values[i];
    }
}


void SimpleNetwork::step() {

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
                s->update_weight();
            });

    this->time_step++;
}

std::vector<float> SimpleNetwork::read_output_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->output_neurons.size());
    for (auto &output_neuro : this->output_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}

std::vector<float> SimpleNetwork::read_all_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->all_neurons.size());
    for (auto &output_neuro : this->all_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}


float SimpleNetwork::introduce_targets(std::vector<float> targets) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neurons[counter]->introduce_targets(targets[counter], this->time_step);
    }
    return error;
}

void SimpleNetwork::initialize_network(const std::vector<std::vector<float>>& input_batch) {
    // calculate mean of inputs and use it to initialize the network
    std::vector<float> mean_of_inputs = mean(input_batch);
    this->set_input_values(mean_of_inputs);

    // turn grads off
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->no_grad = true;
            });

    // pass forward the mean inputs
    std::for_each(
            std::execution::par_unseq,
            input_neurons.begin(),
            input_neurons.end(),
            [&](neuron *n) {
                n->fire(0);
            });

    // parallel execution messes this up
    std::for_each(
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->init_incoming_synapses();
            });

    // turn grads back on
    std::for_each(
            std::execution::par_unseq,
            all_neurons.begin(),
            all_neurons.end(),
            [&](neuron *n) {
                n->no_grad = false;
            });
}
