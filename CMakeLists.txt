#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#GCloud
#set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")

#Cedar
message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")
set(pybind11_DIR $ENV{SLURM_TMPDIR}/CAN/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)
#set(pybind11_DIR "/Users/khurramjaved/miniconda3/envs/ambitious_ml_project/lib/python3.8/site-packages/pybind11/include/pybind11")
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
#set(pybind_find_dir $ENV{SLURM_TMPDIR}/CAN/lib/python3.7/site-packages/pybind11/include/pybind11)
include_directories($ENV{SLURM_TMPDIR}/CAN/lib/python3.7/site-packages/pybind11/include/pybind11)
#include_directories("/Users/khurramjaved/miniconda3/envs/ambitious_ml_project/lib/python3.8/site-packages/pybind11/include/pybind11")


pybind11_add_module(FlexibleNN src/utils.cpp include/utils.h include/nn/synapse.h src/nn/synapse.cpp include/nn/neuron.h
        src/nn/neuron.cpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h
        src/nn/utils.cpp include/nn/message.h src/nn/message.cpp include/nn/networks/feedforward_state_value_network.h
        src/nn/networks/feedforward_state_value_network.cpp include/environments/animal_learning/tracecondioning.h
        src/environments/animal_learning/traceconditioning.cpp tests/include/test_case_networks.h
        tests/src/network_implementations.cpp src/environments/animal_learning/tracepatterning.cpp
        include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp include/nn/networks/recurrent_state_value_network.h
        src/nn/networks/adaptive_recurrent_networks.cpp
        include/nn/networks/network.h src/nn/networks/network.cpp tests/include/feedforward_gradient_testcase.h
        include/nn/networks/expanding_linear_function_approximator.h src/nn/networks/expanding_linear_function_approximator.cpp
        pybind_interface/network_interface.cpp)


# For linking locally
#set(CMAKE_CXX_FLAGS "-O3 \
#-I/home/taodav/Downloads/mariadb-connector-c-3.1.13-ubuntu-bionic-amd64/include/mariadb \
#-I/home/taodav/Downloads/mariadb-connector-c-3.1.13-ubuntu-bionic-amd64/include/mariadb/mysql \
#-L/home/taodav/Downloads/mariadb-connector-c-3.1.13-ubuntu-bionic-amd64/lib/mariadb/ \
#-ltbb -lpthread ")
#include_directories(/usr/local/include/tbb)
#link_directories(/usr/local/lib)

add_executable(TracePatterning src/utils.cpp include/utils.h include/nn/synapse.h src/nn/synapse.cpp include/nn/neuron.h
        src/nn/neuron.cpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        include/nn/message.h src/nn/message.cpp include/nn/networks/feedforward_state_value_network.h
        src/nn/networks/feedforward_state_value_network.cpp include/environments/animal_learning/tracecondioning.h
        src/environments/animal_learning/traceconditioning.cpp tests/include/test_case_networks.h
        tests/src/network_implementations.cpp src/environments/animal_learning/tracepatterning.cpp tracepatterning_exp.cpp
        include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp include/nn/networks/recurrent_state_value_network.h
        src/nn/networks/adaptive_recurrent_networks.cpp
        include/nn/networks/discrete_actions_value_network.h include/nn/networks/network.h src/nn/networks/network.cpp
        tests/include/feedforward_gradient_testcase.h include/nn/generators/feedforward_generator.h tracepatterning_exp.cpp
        tests/src/target_without_sideffects.cpp tests/include/random_data_generator.h tests/src/random_data_generator.cpp)



#Cedar
#target_link_libraries(FlexibleNN PRIVATE mariadb)

# FOR DEBUGGING
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

# For linking locally
target_link_libraries(FlexibleNN PUBLIC tbb)
target_link_libraries(FlexibleNN PUBLIC mariadb)
target_link_libraries(TracePatterning PUBLIC tbb)
target_link_libraries(TracePatterning PUBLIC mariadb)


project(FlexibleNN LANGUAGES CXX)

