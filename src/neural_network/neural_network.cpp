////
//// Created by Khurram Javed on 2021-03-16.
////
//
//#include "../../include/neural_networks/neural_network.h"
//#include <random>
//#include <vector>
//#include "../../include/neural_networks/synapse.h"
//#include <iostream>
//#include <execution>
//#include <algorithm>
//#include <thread>
//#include <mutex>
//
//
//NeuralNetwork::NeuralNetwork(int total_layers, int width) {
////    std::random_device rd;
//    srand(0);
//
//    std::mt19937 mt(0);
//    mt.seed(10);
//    std::uniform_real_distribution<float> dist(-0.01, 0.01);
//
//
//    std::cout << (float(rand()%100))/100 - 0.5 << " " << (float(rand()%100))/100 - 0.5 << std::endl;
//    for(int layer=0; layer<total_layers; layer++)
//    {
////        std::cout << "Layer number " << layer << std::endl;
//        std::vector<neuron*> temp_list;
//        for(int neuron_in_layer=0; neuron_in_layer< width; neuron_in_layer+=1)
//        {
//            auto* n = new neuron();
//
//            temp_list.push_back(n);
//            for(auto & all_neuron : all_neurons)
//            {
//                auto s = new synapse(all_neuron, n, dist(mt));
//                this->all_synapses.push_back(s);
////                this->all_synapses.push_back(synapse(all_neurons[temp_n], n, (float(rand()%100))/100 - 0.5));
//            }
//            if(layer == 0)this->input_neurons.push_back(n);
//            if(layer == total_layers -1)this->output_neuros.push_back(n);
//        }
//        for(auto temp_n : temp_list)
//        {
//            this->all_neurons.push_back(temp_n);
//        }
//    }
//
////    std::cout << "All layers done " << std::endl;
////    std::cout << "Len of matrix = " << this->adjacency_matric.size() << std::endl;
//
//    for(int temp=0; temp<all_neurons.size()-1; temp++)
//    {
//        std::vector<int> vector1(all_neurons.size()-1, -1);
//        this->adjacency_matric.push_back(vector1);
//    }
//
////    std::cout << "Adjacency matrix compute" << std::endl;
//
//    for(auto & all_synapse : this->all_synapses)
//    {
//        this->adjacency_matric[all_synapse->input_neurons->id][all_synapse->output_neurons->id] = 1;
//    }
////    std::cout << "Before index " << all_synapses[0].input_neurons->id << std::endl;
////    std::random_shuffle ( all_synapses.begin(), all_synapses.end() , myrandom);
////    std::cout << "After index " << all_synapses[0].input_neurons->id << std::endl;
//}
//
//
//void NeuralNetwork::delete_network() {
//    for(auto & all_neuron : this->all_neurons)
//    {
//        delete all_neuron;
//    }
//}
//void NeuralNetwork::update_depth_matrix()
//{
//    for(auto & all_synapse : this->all_synapses)
//    {
//        this->adjacency_matric[all_synapse->input_neurons->id][all_synapse->output_neurons->id] = all_synapse->output_neurons->depth;
//    }
//}
//void NeuralNetwork::set_input_values(std::vector<float> const &input_values) {
////    assert(input_values.size() == this->input_neurons.size());
//    for(int i = 0; i<input_values.size(); i++)
//    {
////        std::cout << "Input neuron incoming synapses = " << this->input_neurons[i]->incoming_synapses.size() << " " <<  this->input_neurons[i]->outgoing_synapses.size() <<  std::endl;
//        this->input_neurons[i]->value = input_values[i];
//    }
//}
//
//void NeuralNetwork::step() {
//
//
////    std::for_each(
////            std::execution::par_unseq,
////            all_synapses.begin(),
////            all_synapses.end(),
////            [&](synapse s)
////            {
////                s.step();
////            });
////
////    std::for_each(
////            std::execution::par_unseq,
////            all_neurons.begin(),
////            all_neurons.end(),
////            [&](neuron* n)
////            {
////                n->activation();
////            });
//
//    std::for_each(
//            std::execution::par_unseq,
//            all_neurons.begin(),
//            all_neurons.end(),
//            [&](neuron* n)
//            {
//                n->update_value();
//            });
//
//}
//
//
//std::vector<float> NeuralNetwork::read_output_values() {
//    std::vector<float> output_vec;
//    for(int i=0; i<this->output_neuros.size(); i++)
//    {
//        output_vec.push_back(this->output_neuros[i]->value);
//    }
//    return output_vec;
//}