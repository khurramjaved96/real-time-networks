cmake_minimum_required(VERSION 3.14)
project(benchmarks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread" )

add_executable(benchmarks main.cpp add.cu include/neural_networks/public_header.h add_cpu.cpp simple_nn.cpp src/Experiment.cpp Experiment.h test.cpp src/utils.cpp include/utils.h src/neural_network/neuron.cpp include/neural_networks/neuron.h include/neural_networks/synapse.h src/neural_network/synapse.cpp include/neural_networks/neural_network.h src/neural_network/neural_network.cpp)