//
// Created by Khurram Javed on 2021-04-01.
//

//
// Created by Khurram Javed on 2021-03-30.
//

#include "../../../include/neural_networks/networks/test.h"
#include "../../../include/neural_networks/neuron.h"
#include "../../../include/neural_networks/synapse.h"
#include <assert.h>
#include <random>
#include <execution>
#include <iostream>

TestCase::TestCase(float step_size, int width, int seed) {
    this->time_step = 0;

    int input_neuron = 3;
    for (int counter = 0; counter < input_neuron; counter++) {
        auto n = new neuron(false);
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
    for (int counter=0; counter < output_neuros; counter++)
    {
        auto n = new neuron(false);
        this->output_neuros.push_back(n);
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

    for(auto it: this->all_synapses){
        sum_of_gradients.push_back(0);
    }


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

//}

void TestCase::print_graph(neuron *root) {

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


void TestCase::set_print_bool() {
    std::cout
            << "From\tTo\tGrad_queue_size\tFrom activations_size\tTo activations_size\tError grad_queue From\tCredit\n";
    for (auto &s : this->all_synapses)
        s->print_status = false;
}


int TestCase::get_input_size() {
    return this->input_neurons.size();
}

int TestCase::
get_total_synapses() {
    return this->all_synapses.size();
}

TestCase::~TestCase() {
    for (auto &it : this->all_neurons)
        delete it;
    for (auto &it : this->all_synapses)
        delete it;
}


void TestCase::set_input_values(std::vector<float> const &input_values) {
    assert(input_values.size() == this->input_neurons.size());
    for (int i = 0; i < input_values.size(); i++) {
        this->input_neurons[i]->temp_value = input_values[i];
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
                n->propogate_error();
            });

    for(int counter = 0; counter < all_synapses.size(); counter ++)
    {
        this->sum_of_gradients[counter] += all_synapses[counter]->credit;
    }
//    std::for_each(
//            std::execution::par_unseq,
//            all_synapses.begin(),
//            all_synapses.end(),
//            [&](synapse *s) {
//                s->update_weight();
//            });


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
//            output_neurons.begin(),
//            output_neurons.end(),
//            [&](neuron* n)
//            {
//                n->update_value();
//            });

}

std::vector<float> TestCase::read_output_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->output_neuros.size());
    for (auto &output_neuro : this->output_neuros) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}

std::vector<float> TestCase::read_all_values() {
    std::vector<float> output_vec;
    output_vec.reserve(this->all_neurons.size());
    for (auto &output_neuro : this->all_neurons) {
        output_vec.push_back(output_neuro->value);
    }
    return output_vec;
}


float TestCase::introduce_targets(std::vector<float> targets) {
    float error = 0;
    for (int counter = 0; counter < targets.size(); counter++) {
        error += this->output_neuros[counter]->introduce_targets(targets[counter], this->time_step);
    }
    return error;
}