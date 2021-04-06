//
// Created by Khurram Javed on 2021-03-16.
//

#ifndef BENCHMARKS_NEURON_H
#define BENCHMARKS_NEURON_H



#include <vector>
#include <queue>
#include <mutex>
#include <thrust/device_vector.h>


class neuron {
public:
    static int neuron_id;
    float value;
    float temp_value;
    int depth;
    int id;
//    thrust::device_vector<neuron> parents;
//    std::mutex value_mutex;
//    std::mutex depth_mutex;
//    std::queue<float> error_gradient;
//    std::queue<float> past_activations;
//    std::vector<std::reference_wrapper<neuron>> input_nodes;
//    std::vector<std::reference_wrapper<neuron>> output_nodes;
//    std::vector<std::reference_wrapper<float>> input_weights;
//    std::vector<std::reference_wrapper<float>> output_weights;

    neuron();
    void activation();
};

#endif //BENCHMARKS_NEURON_H