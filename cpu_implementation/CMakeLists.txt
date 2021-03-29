cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX CUDA)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread" )
set(INCLUDE_DIRS "../extern/pybind11/include")
#include_directories(${/project/def-whitem/khurram/miniconda3/include/})
add_library(FlexibleNN SHARED simple_nn.cpp src/utils.cpp include/utils.h include/neural_networks/neuron.h include/neural_networks/synapse.h src/neural_network/neuron.cpp  src/neural_network/synapse.cpp include/neural_networks/neural_network.h src/neural_network/neural_network.cpp)
target_include_directories(FlexibleNN PRIVATE ${INCLUDE_DIRS})
#target_link_libraries(FlexibleNN PRIVATE Binding)
#add_executable(neural_network simple_nn.cpp src/utils.cpp include/utils.h include/neural_networks/neuron.h include/neural_networks/synapse.h src/neural_network/neuron.cpp  src/neural_network/synapse.cpp include/neural_networks/neural_network.h src/neural_network/neural_network.cpp)
