//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/neural_networks/neuron.h"
#include <iostream>

neuron::neuron() {
    value = 0;
    depth = 1;
    temp_value = 0;
    id = neuron_id;
    neuron_id++;
}


void neuron::activation() {
    if (this->temp_value > 0) this->value = this->temp_value;
    else this->value = 0;

    this->temp_value = 0;
//    this->past_activations.push(this->value);
}

int neuron::neuron_id = 0;