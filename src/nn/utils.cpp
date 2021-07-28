//
// Created by Khurram Javed on 2021-03-30.
//
#include "../../include/nn/utils.h"
#include <assert.h>
#include <cmath>
#include <vector>

#include <algorithm>
#include <random>
#include <iostream>

#include "../../include/nn/synapse.h"
#include "../../include/nn/neuron.h"
#include "../../include/nn/dynamic_elem.h"

float sigmoid(float a) {
  return (1.0) / (1.0 + exp(-1 * a));
}

float relu(float a) {
  if (a > 0) return a;
  return 0;
}

template<class mytype>
mytype max(std::vector<mytype> values) {
  auto it = std::max_element(values.begin(), values.end());
  return *it;
}

bool is_null_ptr(dynamic_elem *elem) {
//    return true;
  if (elem == nullptr)
    return true;
  return false;
}

bool to_delete_s(synapse *s) {
  return s->is_useless;
}

bool to_delete_n(Neuron *s) {
  return s->useless_neuron;
}

std::vector<float> one_hot_encode(int no, int total) {
  std::vector<float> my_vec;
  my_vec.reserve(total);
  for (int t = 0; t < total; t++) {
    my_vec.push_back(0);
  }
  my_vec[no] = 1;
  return my_vec;
}

std::vector<float> sum(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  std::vector<float> ans_vector;
  ans_vector.reserve(lhs.size());
  assert(lhs.size() == rhs.size());
  for (int counter = 0; counter < lhs.size(); counter++) {
    ans_vector.push_back(lhs[counter] + rhs[counter]);
  }
  return ans_vector;
}

std::vector<int> sum(const std::vector<int> &lhs, const std::vector<int> &rhs) {
  std::vector<int> ans_vector;
  ans_vector.reserve(lhs.size());
  assert(lhs.size() == rhs.size());
  for (int counter = 0; counter < lhs.size(); counter++) {
    ans_vector.push_back(lhs[counter] + rhs[counter]);
  }
  return ans_vector;
}

float max(std::vector<float> values) {
  auto it = std::max_element(values.begin(), values.end());
  return *it;
}

float min(std::vector<float> values) {
  auto it = std::min_element(values.begin(), values.end());
  return *it;
}

float min(float a, float b) {
  if (a < b)
    return a;
  return b;
}

float max(float a, float b) {
  if (a > b)
    return a;
  return b;
}

std::vector<float> softmax(const std::vector<float> &values) {
  float max_val = max(values);
  std::vector<float> results;
  float den = 0;
  for (auto &it : values) {
    den += exp(it - max_val);
  }
  results.reserve(values.size());
  for (auto &it : values) {
    results.push_back(exp(it - max_val) / den);
  }
  return results;
}

float sum(const std::vector<float> &values) {
  float total = 0;
  for (auto &it : values)
    total += it;
  return total;
}

std::vector<float> mean(const std::vector<std::vector<float>> &values) {
  // returns mean for each dim
  int dim = values[0].size();
  std::vector<float> ans_vector(dim, 0);
  for (const auto &value : values)
    ans_vector = sum(ans_vector, value);
  for (int counter = 0; counter < dim; counter++)
    ans_vector[counter] = ans_vector[counter] / values.size();
  return ans_vector;
}

uniform_random::uniform_random(int seed) {
  this->mt = std::mt19937(seed);
  this->dist = std::uniform_real_distribution<float>(-1, 1);
}

std::vector<float> uniform_random::get_random_vector(int size) {
  std::vector<float> output;
  output.reserve(size);
  for (int counter = 0; counter < size; counter++)
    output.push_back(dist(mt));
  return output;
}

normal_random::normal_random(int seed, float mean, float stddev) {
  this->mt = std::mt19937(seed);
  this->dist = std::normal_distribution<float>(mean, stddev);
}

float normal_random::get_random_number() {
  float value = dist(mt);
  return dist(mt);
}

std::vector<float> normal_random::get_random_vector(int size) {
  std::vector<float> output;
  output.reserve(size);
  for (int counter = 0; counter < size; counter++)
    output.push_back(dist(mt));
  return output;
}
