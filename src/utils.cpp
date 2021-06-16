//
// Created by Khurram Javed on 2021-03-16.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "../include/utils.h"


void print_vector(std::vector<float> const &v) {
    std::cout << "[";
    int counter = 0;
    for (auto i = v.begin(); i != v.end(); ++i) {
        std::cout << " " << std::setw(3)  <<  *i << ",";
        if (counter % 50 == 49) std::cout << "\n";
        counter++;
    }
    //std::cout << "]\n";
    std::cout << " ]";
}

void print_vector(std::vector<int> const &v) {
    std::cout << "[";
    int counter = 0;
    for (auto i = v.begin(); i != v.end(); ++i) {
        std::cout << " "  << std::setw(3) << *i << ",";
        if (counter % 100 == 99) std::cout << "\n";
        counter++;
    }
    //std::cout << "]\n";
    std::cout << " ]";
}


void print_vector(std::vector<bool> const &v) {
    std::cout << "[";
    int counter = 0;
    for (auto i = v.begin(); i != v.end(); ++i) {
        std::cout << " "  << std::setw(3) << *i << ",";
        if (counter % 100 == 99) std::cout << "\n";
        counter++;
    }
    //std::cout << "]\n";
    std::cout << " ]";
}

void print_matrix(std::vector<std::vector<int>> const &v) {
    int counter = 0;
    for(int temp=0; temp<v.size(); temp++) {
        if (temp > 50)
        {   std::cout << "Truncating output\n";
            break;
        }
        std::cout << "[";
        int counter_inner = 0;
        for (auto i = v[temp].begin(); i != v[temp].end(); ++i) {
            if(counter_inner > 50){
                std::cout << ", ... , ]";
                break;
            }
            std::cout << " " << std::setw(3)  << *i << ",";
//            if (counter % 50 == 49) std::cout << "\n";
            counter++;
            counter_inner++;
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

NetworkVisualizer::NetworkVisualizer(std::vector<neuron *> all_neurons) {
    this->all_neurons = all_neurons;
}

NetworkVisualizer::NetworkVisualizer(std::vector<neuron *> all_neurons, std::vector<no_grad_synapse *> no_grad_synapses) {
    this->all_neurons = all_neurons;
    this->no_grad_synapses = no_grad_synapses;
}
void NetworkVisualizer::generate_dot(int time_step) {
    std::string edge_color;
    dot_string = "digraph network{\n"
                 "\tedge [fontsize=8];\n"
                 "\tnode [shape = circle];\n";

    for(auto &it : all_neurons) {
        for (auto &os: it->outgoing_synapses) {
            if (os->input_neuron->mature and os->output_neuron->mature) {
                auto current_n = os;
                dot_string += "\t" + std::to_string(current_n->input_neuron->id)
                              + "->" + std::to_string(current_n->output_neuron->id) //+ ";\n";
                              + "[label = \"" + std::to_string(os->weight) + "\"];\n";
            }
        }
    }
//    for (auto &os: this->no_grad_synapses) {
//        auto current_n = os;
//        dot_string += "\t" + std::to_string(current_n->input_neuron->id)
//                      + "->" + std::to_string(current_n->output_neuron->id) //+ ";\n";
//                      + "[style=dashed];\n";
//    }
    dot_string += "\n}";
    std::ofstream dot_file("vis/" + std::to_string(time_step) + "_simple" + ".gv");
    dot_file << dot_string;
    dot_file.close();
}

std::string NetworkVisualizer::get_graph(int time_step) {
    std::string edge_color;
    dot_string = "digraph network{\n"
                 "\tedge [fontsize=8];\n"
                 "\tnode [shape = circle];\n";

    for(auto &it : all_neurons) {
        for (auto &os: it->outgoing_synapses) {
            if (os->input_neuron->mature and (os->output_neuron->mature or os->output_neuron->is_output_neuron)) {
              auto current_n = os;
              fabs(os->weight) < 0.0005 ? edge_color="darkgrey" : edge_color="black";
              if (current_n->pass_gradients)
                  dot_string += "\t" + std::to_string(current_n->input_neuron->id)
                                + "->" + std::to_string(current_n->output_neuron->id) //+ ";\n";
                                + "[label = \"" + std::to_string(os->weight) + "\", color=" + edge_color + "];\n";
              else
                  dot_string += "\t" + std::to_string(current_n->input_neuron->id)
                                + "->" + std::to_string(current_n->output_neuron->id) //+ ";\n";
                                + "[label = \"" + std::to_string(os->weight) + "\", style=dashed, color=red];\n";
            }
        }
    }
    dot_string += "\n}";
    return dot_string;
}

std::string NetworkVisualizer::get_graph_detailed(int time_step) {
    std::string edge_color;
    dot_string = "digraph network{\n"
                 "\tconcentrate=True;\n"
                 "\trankdir=TB;\n"
                 "\tnode [shape=record];";

    for(auto &it : all_neurons) {
        dot_string += "\t" + std::to_string(it->id)
                      + " [label=\"" + std::to_string(it->id) + "\\n|"
                      + "{past_act_size:|err_grad_size:|value:}"
                      + "|{" + std::to_string(it->past_activations.size()) +"|"
                      +  std::to_string(it->error_gradient.size()) + "|"
                      +  std::to_string(it->value) +"}"
                      + "\"];\n";
        for (auto &os: it->outgoing_synapses) {
            auto current_n = os;
            dot_string += "\t" + std::to_string(current_n->input_neuron->id)
                          + "->" + std::to_string(current_n->output_neuron->id) + ";\n";
            //+ " [label = \"gq_size:" + std::to_string(os->grad_queue.size())
            //+ " credit:" + std::to_string(os->credit) + "\"];\n";

            // + "[label = \"" + std::to_string(os->weight) + "\"];\n";
        }
    }
//    for (auto &os: this->no_grad_synapses) {
//        auto current_n = os;
//        dot_string += "\t" + std::to_string(current_n->input_neurons->id)
//                      + "->" + std::to_string(current_n->output_neurons->id) //+ ";\n";
//                      + "[style=dashed];\n";
//    }
    dot_string += "\n}";
    return dot_string;
}


void NetworkVisualizer::generate_dot_detailed(int time_step) {
    std::string edge_color;
    dot_string = "digraph network{\n"
                 "\tconcentrate=True;\n"
                 "\trankdir=TB;\n"
                 "\tnode [shape=record];";

    for(auto &it : all_neurons) {
        dot_string += "\t" + std::to_string(it->id)
                      + " [label=\"" + std::to_string(it->id) + "\\n|"
                      + "{past_act_size:|err_grad_size:|value:}"
                      + "|{" + std::to_string(it->past_activations.size()) +"|"
                      +  std::to_string(it->error_gradient.size()) + "|"
                      +  std::to_string(it->value) +"}"
                      + "\"];\n";
        for (auto &os: it->outgoing_synapses) {
            auto current_n = os;
            dot_string += "\t" + std::to_string(current_n->input_neuron->id)
                          + "->" + std::to_string(current_n->output_neuron->id) + ";\n";
                          //+ " [label = \"gq_size:" + std::to_string(os->grad_queue.size())
                          //+ " credit:" + std::to_string(os->credit) + "\"];\n";

                         // + "[label = \"" + std::to_string(os->weight) + "\"];\n";
        }
    }
//    for (auto &os: this->no_grad_synapses) {
//        auto current_n = os;
//        dot_string += "\t" + std::to_string(current_n->input_neurons->id)
//                      + "->" + std::to_string(current_n->output_neurons->id) //+ ";\n";
//                      + "[style=dashed];\n";
//    }
    dot_string += "\n}";
    std::ofstream dot_file("vis/" + std::to_string(time_step) + "_detailed" + ".gv");
    dot_file << dot_string;
    dot_file.close();
}
