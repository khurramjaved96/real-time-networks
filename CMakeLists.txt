cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#GCloud
set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")

#Cedar
#set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")

# For linking locally
#set(CMAKE_CXX_FLAGS "-O3 \
#-I/home/taodav/Downloads/mariadb-connector-c-3.1.13-ubuntu-bionic-amd64/include/mariadb \
#-I/home/taodav/Downloads/mariadb-connector-c-3.1.13-ubuntu-bionic-amd64/include/mariadb/mysql \
#-L/home/taodav/Downloads/mariadb-connector-c-3.1.13-ubuntu-bionic-amd64/lib/mariadb/ \
#-ltbb -lpthread ")
#include_directories(/usr/local/include/tbb)
#link_directories(/usr/local/lib)

add_executable(FlexibleNN src/utils.cpp include/utils.h include/neural_networks/synapse.h src/neural_network/synapse.cpp include/neural_networks/neuron.h src/neural_network/neuron.cpp include/neural_networks/neural_network.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp  include/neural_networks/utils.h src/neural_network/utils.cpp include/neural_networks/message.h src/neural_network/message.cpp include/neural_networks/networks/adaptive_network.h src/neural_network/networks/adaptive_network.cpp include/animal_learning/tracecondioning.h src/animal_learning/traceconditioning.cpp include/neural_networks/networks/test.h src/neural_network/networks/test.cpp src/animal_learning/tracepatterning.cpp tracepatterning_exp.cpp include/neural_networks/dynamic_elem.h src/neural_network/dynamic_elem.cpp include/neural_networks/networks/adaptive_recurrent_network.h src/neural_network/networks/adaptive_recurrent_networks.cpp recurrence_test.cpp src/neural_network/networks/test_recurrent.cpp include/neural_networks/networks/test_recurrent.h)


#Cedar
#target_link_libraries(FlexibleNN PRIVATE mariadb)

# FOR DEBUGGING
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

# For linking locally
target_link_libraries(FlexibleNN PUBLIC tbb)
target_link_libraries(FlexibleNN PUBLIC mariadb)

project(FlexibleNN LANGUAGES CXX)

