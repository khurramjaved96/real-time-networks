//
// Created by Khurram Javed on 2021-03-30.
//

#include <vector>
#include <cmath>
#include "../../include/neural_networks/utils.h"
#include <algorithm>
#include <random>
#include <assert.h>

float sigmoid(float a){
    return 1/(1+exp(-1*a));
}
float relu(float a){
    if(a > 0) return a;
    return 0;
}
template <class mytype>
mytype max(std::vector<mytype> values){
    auto it = std::max_element(values.begin(), values.end());
    return *it;
}
//

std::vector<float> sum(const std::vector<float> &lhs, const std::vector<float> &rhs){
    std::vector<float> ans_vector;
    ans_vector.reserve(lhs.size());
    assert(lhs.size() == rhs.size());
    for(int counter = 0; counter < lhs.size(); counter++)
    {
        ans_vector.push_back(lhs[counter] + rhs[counter]);
    }
    return ans_vector;
}

std::vector<int> sum(const std::vector<int> &lhs, const std::vector<int> &rhs){
    std::vector<int> ans_vector;
    ans_vector.reserve(lhs.size());
    assert(lhs.size() == rhs.size());
    for(int counter = 0; counter < lhs.size(); counter++)
    {
        ans_vector.push_back(lhs[counter] + rhs[counter]);
    }
    return ans_vector;
}

float max(std::vector<float> values){
    auto it = std::max_element(values.begin(), values.end());
    return *it;
}
float min(std::vector<float> values){
    auto it = std::min_element(values.begin(), values.end());
    return *it;
}
std::vector<float> softmax(const std::vector<float>& values){
    float max_val = max(values);
    std::vector<float> results;
    float den=0;
    for(auto &it : values){
        den += exp(it - max_val);
    }
    results.reserve(values.size());
    for(auto &it : values){
        results.push_back(exp(it - max_val)/den);
    }
    return results;
}

float sum(const std::vector<float>& values){
    float total = 0;
    for(auto &it : values)
        total += it;
    return total;
}

uniform_random::uniform_random(int seed) {
    this->mt = std::mt19937(seed);
    this->dist = std::uniform_real_distribution<float>(-1, 1);
}


std::vector<float> uniform_random::get_random_vector(int size){
    std::vector<float> output;
    output.reserve(size);
    for(int counter=0; counter<size; counter++)
        output.push_back(dist(mt));
    return output;
}
