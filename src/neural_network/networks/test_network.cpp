//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//

#include "../../../include/neural_networks/networks/test_network.h"
#include "../../../include/neural_networks/neuron.h"
#include "../../../include/neural_networks/synapse.h"
#include <assert.h>
#include <random>
#include <execution>
#include <iostream>

CustomNetwork::CustomNetwork(float step_size, int width, int seed) {
    this->time_step = 0;
//    for (int counter = 0; counter < width; counter++) {
//        auto n = new neuron((counter < width - 2));
//        if (counter < 2) {
//            this->input_neurons.push_back(n);
//        }
//        this->all_neurons.push_back(n);
//        if (counter == width - 1 || counter == width - 2) {
//            this->output_neuros.push_back(n);
//        }
//    }
    int input_neuron = 10;
    for (int counter = 0; counter < input_neuron; counter++) {
        auto n = new neuron(true);
        this->input_neurons.push_back(n);
        this->all_neurons.push_back(n);
    }

    int output_neuros = 1;
    for (int counter=0; counter < output_neuros; counter++)
    {
        auto n = new neuron(false);
        this->output_neuros.push_back(n);
        this->all_neurons.push_back(n);
    }

    std::mt19937 mt(seed);
    mt.seed(seed);
    std::uniform_real_distribution<float> dist(-0.001, 0.001);

//    this->all_synapses.push_back(new synapse(all_neurons[0], all_neurons[499], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[0], all_neurons[499], dist(mt), step_size));
    std::vector<neuron*> neurons_so_far;
    for(int layer=0; layer < 5; layer++)
    {
        std::vector<neuron*> this_layer;
        for(int this_layer_neuron = 0; this_layer_neuron < width; this_layer_neuron++) {
            auto* n = new neuron(true);
            this->all_neurons.push_back(n);
            this_layer.push_back(n);
//            adding connections from input
            for(auto &it : this->input_neurons){
                this->all_synapses.push_back(new synapse(it, n, dist(mt), step_size));
            }
            for(auto &it : this->output_neuros){
                this->all_synapses.push_back(new synapse(n, it,  dist(mt), step_size));
            }
            for (auto &it : neurons_so_far) {
                this->all_synapses.push_back(new synapse(it, n, dist(mt), step_size));
            }
        }
        for(auto &it : this_layer){
            neurons_so_far.push_back(it);
        }
//        for (int counter = 2; counter < width; counter++) {
//
//            this->all_synapses.push_back(new synapse(all_neurons[0], all_neurons[counter], dist(mt), step_size));
//            this->all_synapses.push_back(new synapse(all_neurons[1], all_neurons[counter], dist(mt), step_size));
//        }
//        for (int counter = 2; counter < width - 2; counter++) {
//            this->all_synapses.push_back(new synapse(all_neurons[counter], all_neurons[width - 1], dist(mt), step_size));
//            this->all_synapses.push_back(new synapse(all_neurons[counter], all_neurons[width - 2], dist(mt), step_size));
//        }
    }

//    this->all_synapses.push_back(new synapse(all_neurons[1], all_neurons[6], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[1], all_neurons[2], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[2], all_neurons[3], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[1], all_neurons[3], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[2], all_neurons[5], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[0], all_neurons[4], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[3], all_neurons[4], dist(mt), step_size));
//
//    this->all_synapses.push_back(new synapse(all_neurons[4], all_neurons[5], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[5], all_neurons[6], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[2], all_neurons[6], dist(mt), step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[3], all_neurons[6], dist(mt), step_size));

}

void CustomNetwork::print_graph(neuron *root) {

    for (auto &os: root->outgoing_synapses) {
        auto current_n = os;

        if (!current_n->print_status) {

            std::cout << current_n->input_neurons->id << "\t" << current_n->output_neurons->id << "\t"
                      << os->grad_queue.size() << "\t\t" << current_n->input_neurons->past_activations.size()
                      << "\t\t\t" << current_n->output_neurons->past_activations.size() << "\t\t\t"
                      << current_n->input_neurons->error_gradient.size()
                      << "\t\t" << current_n->credit << std::endl;
            current_n->print_status = true;
        }
        print_graph(current_n->output_neurons);
    }
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
    assert(input_values.size() == this->input_neurons.size());
    for (int i = 0; i < input_values.size(); i++) {
        this->input_neurons[i]->temp_value = input_values[i];
    }
}

void CustomNetwork::step() {


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


    //    std::cout << "Neurons firing\n";
//    for(auto& it : this->all_neurons){
//        it->fire(this->time_step);
//    }
//
////    std::cout << "Neurons updating value\n";
//    for(auto& it : this->all_neurons){
//        it->update_value();
//    }
////    std::cout << "Neurons forwarding gradients to synapses\n";
//
//    for(auto& it : this->all_neurons){
//        it->forward_gradients();
//    }
//
////    std::cout << "Propagating gradients from synapses to neuron (Summing with correct time-steps)\n";
//
//    for(auto& it : this->all_neurons){
//        it->propogate_error();
//    }
//
//    for(auto& it: this->all_synapses){
//        it->update_weight();
////        it->zero_gradient();
//    }


    this->time_step++;





//
//    std::for_each(
//            std::execution::par_unseq,
//            output_neuros.begin(),
//            output_neuros.end(),
//            [&](neuron* n)
//            {
//                n->update_value();
//            });

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
        error += this->output_neuros[counter]->introduce_targets(targets[counter], this->time_step);
    }
    return error;
}