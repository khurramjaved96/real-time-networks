//
// Created by taodav on 24/6/21.
//

#include "../../../include/neural_networks/networks/test_add_delete.h"
#include "../../../include/neural_networks/synapse.h"
#include "../../../include/neural_networks/networks/adaptive_network.h"


TestAddDelete::TestAddDelete(float step_size, int width, int seed) : TestCase() {
    this->time_step = 0;

    int input_neuron = 3;
    for (int counter = 0; counter < input_neuron; counter++) {
        auto n = new neuron(false, false, true);
        this->input_neurons.push_back(n);
        this->all_neurons.push_back(n);
    }


    bool relu = true;
    auto n = new neuron(relu, false);
    this->all_neurons.push_back(n);

    n = new neuron(relu, false);
    this->all_neurons.push_back(n);

    n = new neuron(relu, false);
    this->all_neurons.push_back(n);

    int output_neurons = 1;
    for (int counter=0; counter < output_neurons; counter++)
    {
        auto n = new neuron(false, true);
        this->output_neurons.push_back(n);
        this->all_neurons.push_back(n);
    }

    this->all_synapses.push_back(new synapse(all_neurons[1 - 1], all_neurons[4 - 1], 0.2, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[1 - 1], all_neurons[6 - 1], 0.5, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[4 - 1], -0.2, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[2 - 1], all_neurons[5 - 1], 0.7, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[4 - 1], 0.65, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[5 - 1], 0.1, step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[3 - 1], all_neurons[7 - 1], -0.1, step_size));
//    this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[6 - 1], 0.2, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[4 - 1], all_neurons[7 - 1], -0.1, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[5 - 1], all_neurons[7 - 1], 0.2, step_size));
    this->all_synapses.push_back(new synapse(all_neurons[6 - 1], all_neurons[7 - 1], 0.2, step_size));

    for(auto it: this->all_synapses){
        sum_of_gradients.push_back(0);
    }

}

/**
 * We add a fixed feature here.
 * @param step_size
 * @return
 */
void TestAddDelete::add_feature(float step_size) {
    neuron *last_neuron = new neuron(true);
    this->all_neurons.push_back(last_neuron);

//  Attach this neuron to all input neurons
    for (auto &n : this->input_neurons) {
        auto syn = new synapse(n, last_neuron, 0.01, step_size);
        syn->block_gradients();
        this->all_synapses.push_back(syn);
        this->sum_of_gradients.push_back(0);
    }

//  Attach this neuron to all output neurons
    for (auto &output_n : this->output_neurons) {
        synapse *output_s_temp = new synapse(last_neuron, output_n, 1, 0);
        this->all_synapses.push_back(output_s_temp);
        this->sum_of_gradients.push_back(0);
    }
}

void TestAddDelete::delete_feature() {
//  We delete neuron all_neurons[6 - 1] here
    this->all_synapses[1]->useless = true;
    this->all_neurons[6 - 1]->useless_neuron = true;
    this->all_synapses[8]->useless = true;

    auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_s);
    this->all_synapses.erase(it, this->all_synapses.end());
    this->sum_of_gradients.erase(this->sum_of_gradients.begin() + 1);
    this->sum_of_gradients.erase(this->sum_of_gradients.begin() + 8 - 1);

    auto it_n = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_n);
    this->all_neurons.erase(it_n, this->all_neurons.end());
}
