//
// Created by Khurram Javed on 2021-03-30.
//

#ifndef FLEXIBLENN_UTILS_H
#define FLEXIBLENN_UTILS_H

#include <vector>
#include <random>

float sigmoid(float a);
float relu(float a);
template <class mytype>
mytype max(std::vector<mytype> values);
float max(std::vector<float> values);
float min(std::vector<float> values);
float min(float, float);
float max(float, float);
std::vector<float> softmax(const std::vector<float>&);
float sum(const std::vector<float>&);
std::vector<float> sum(std::vector<float> &lhs, std::vector<float> &rhs);
std::vector<int> sum(std::vector<int> &lhs, std::vector<int> &rhs);
int argmax(std::vector<float>);
std::vector<float> mean(const std::vector<std::vector<float>>&);
std::vector<float> one_hot_encode(int no, int total_numbers);
class uniform_random{
    std::mt19937 mt;
    std::uniform_real_distribution<float> dist;
public:
    uniform_random(int seed);
    std::vector<float> get_random_vector(int size);
};

class normal_random{
    std::mt19937 mt;
    std::normal_distribution<float> dist;
public:
    normal_random(int seed, float mean, float stddev);
    float get_random_number();
    std::vector<float> get_random_vector(int size);
};
#endif //FLEXIBLENN_UTILS_H
//