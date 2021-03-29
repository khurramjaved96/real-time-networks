//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/neural_networks/neural_network.h"
#include <random>
#include <vector>
#include "../../include/neural_networks/synapse.h"
#include <iostream>
#include <execution>
#include <algorithm>
#include <thread>
#include <mutex>

std::mutex neuron_mutex;
int myrandom (int i) { return std::rand()%i;}


NeuralNetwork::NeuralNetwork(int total_layers, int width) {
//    std::random_device rd;
    srand(0);

    std::mt19937 mt(0);
    mt.seed(10);
    std::uniform_real_distribution<float> dist(-0.0001, 0.0001);



    std::cout << (float(rand()%100))/100 - 0.5 << " " << (float(rand()%100))/100 - 0.5 << std::endl;
    for(int layer=0; layer<total_layers; layer++)
    {
        std::vector<neuron*> temp_list;
        for(int neuron_in_layer=0; neuron_in_layer< width; neuron_in_layer+=1)
        {
            neuron* n = new neuron();

            temp_list.push_back(n);
            for(int temp_n = 0; temp_n<all_neurons.size(); temp_n++)
            {
                this->all_synapses.push_back(synapse(all_neurons[temp_n], n, dist(mt)));
//                this->all_synapses.push_back(synapse(all_neurons[temp_n], n, (float(rand()%100))/100 - 0.5));
            }
            if(layer == 0)this->input_neurons.push_back(n);
            if(layer == total_layers -1)this->output_neuros.push_back(n);
        }
        for(int temp_n =0; temp_n < temp_list.size(); temp_n++)
        {
            this->all_neurons.push_back(temp_list[temp_n]);
        }
    }


    for(int temp=0; temp<all_neurons.size()-1; temp++)
    {
        std::vector<int> vector1(all_neurons.size()-1, -1);
        this->adjacency_matric.push_back(vector1);
    }

    for(int synapse_id = 0; synapse_id < this->all_synapses.size(); synapse_id++)
    {
        this->adjacency_matric[this->all_synapses[synapse_id].input_neurons->id][this->all_synapses[synapse_id].output_neurons->id] = 1;
    }
//    std::cout << "Before index " << all_synapses[0].input_neurons->id << std::endl;
//    std::random_shuffle ( all_synapses.begin(), all_synapses.end() , myrandom);
//    std::cout << "After index " << all_synapses[0].input_neurons->id << std::endl;
}

void NeuralNetwork::update_depth_matrix()
{
    for(int synapse_id = 0; synapse_id < this->all_synapses.size(); synapse_id++)
    {
        this->adjacency_matric[this->all_synapses[synapse_id].input_neurons->id][this->all_synapses[synapse_id].output_neurons->id] = this->all_synapses[synapse_id].output_neurons->depth;
    }
}
void NeuralNetwork::set_input_values(std::vector<float> const &input_values) {
//    assert(input_values.size() == this->input_neurons.size());
    for(int i = 0; i<input_values.size(); i++)
    {
        this->input_neurons[i]->value = input_values[i];
    }
}

void NeuralNetwork::step() {

//    for(int i = all_synapses.size()-1; i > -1; i--)
//    {
//        all_synapses[i].step();
//    }
//    for(int i = 0; i < all_synapses.size(); i++)
//    {
//        all_synapses[i].step();
//    }

    std::for_each(
            std::execution::par_unseq,
            all_synapses.begin(),
            all_synapses.end(),
            [&](synapse s)
            {
                s.step();
            });

//    std::for_each(
//            std::execution::par_unseq,
//            all_neurons.begin(),
//            all_neurons.end(),
//            [&](neuron* n)
//            {
//                n->activation();
//            });

}

void process(synapse s)
{
//    sstep();
}

std::vector<float> NeuralNetwork::read_output_values() {
    std::vector<float> output_vec;
    for(int i=0; i<this->output_neuros.size(); i++)
    {
        output_vec.push_back(this->output_neuros[i]->value);
    }
    return output_vec;
}