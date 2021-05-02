cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread" )

add_executable(FlexibleNN src/utils.cpp include/utils.h include/neural_networks/synapse.h   src/neural_network/synapse.cpp include/neural_networks/neuron.h src/neural_network/neuron.cpp  include/neural_networks/neural_network.h src/neural_network/neural_network.cpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/neural_networks/networks/linear.h include/neural_networks/utils.h src/neural_network/utils.cpp src/neural_network/networks/linear.cpp include/neural_networks/message.h src/neural_network/message.cpp include/neural_networks/networks/test_network.h src/neural_network/networks/test_network.cpp simple_exp.cu src/hybrid_code/queue.cu src/host_code/queue.cpp src/host_code/device_queue.h include/animal_learning/tracecondioning.h src/animal_learning/traceconditioning.cpp test_case.cpp include/neural_networks/networks/test.h src/neural_network/networks/test.cpp)

target_link_libraries(FlexibleNN PRIVATE mariadb)


#CMAKE OLD BOILDERPLATE CODE FOR INTERACGING C++ WITH PYTHON.
#cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)
#
#
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread" )
##set(INCLUDE_DIRS "../extern/pybind11/include")
#add_subdirectory("../extern/pybind11" "./pybind")
##set(PTHON_DEV_TOOLS_DIRECTORY "/project/def-whitem/khurram/miniconda3/include/python3.7m/")
##include_directories(${/project/def-whitem/khurram/miniconda3/include/})
#pybind11_add_module(FlexibleNN SHARED simple_nn.cpp src/utils.cpp include/utils.h include/neural_networks/neuron.h include/neural_networks/synapse.h src/neural_network/neuron.cpp  src/neural_network/synapse.cpp include/neural_networks/neural_network.h src/neural_network/neural_network.cpp)
##target_include_directories(FlexibleNN PRIVATE ${INCLUDE_DIRS})
#target_include_directories(FlexibleNN PRIVATE ${PTHON_DEV_TOOLS_DIRECTORY})
##target_link_libraries(FlexibleNN PRIVATE Binding)
##add_executable(neural_network simple_nn.cpp src/utils.cpp include/utils.h include/neural_networks/neuron.h include/neural_networks/synapse.h src/neural_network/neuron.cpp  src/neural_network/synapse.cpp include/neural_networks/neural_network.h src/neural_network/neural_network.cpp)
##