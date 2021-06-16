cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-O0 -ltbb -lpthread" )

#add_executable(FlexibleNN src/utils.cpp include/utils.h include/neural_networks/utils.h src/neural_network/utils.cpp include/neural_networks/synapse.h   src/neural_network/synapse.cpp include/neural_networks/neuron.h src/neural_network/neuron.cpp  include/neural_networks/neural_network.h src/neural_network/neural_network.cpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/neural_networks/message.h src/neural_network/message.cpp include/neural_networks/networks/test_network.h src/neural_network/networks/test_network.cpp run_tmaze.cu include/neural_networks/networks/test.h src/neural_network/networks/test.cpp src/hybrid_code/queue.cu include/animal_learning/tracecondioning.h src/animal_learning/traceconditioning.cpp include/neural_networks/networks/simple_network.h src/neural_network/networks/simple_network.cpp src/environments/tmaze.cpp include/environments/tmaze.h)
set(CMAKE_CXX_STANDARD 20)

#GCloud
#set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")

#Cedar
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")

#old tmaze
#add_executable(FlexibleNN simple_exp.cpp src/utils.cpp include/utils.h include/neural_networks/synapse.h src/neural_network/synapse.cpp include/neural_networks/neuron.h src/neural_network/neuron.cpp include/neural_networks/neural_network.h src/neural_network/neural_network.cpp include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp  include/neural_networks/utils.h src/neural_network/utils.cpp include/neural_networks/message.h src/neural_network/message.cpp include/neural_networks/networks/adaptive_network.h src/neural_network/networks/adaptive_network.cpp run_tmaze.cpp src/environments/tmaze.cpp include/environments/tmaze.h)
add_executable(FlexibleNN src/utils.cpp include/utils.h include/neural_networks/synapse.h src/neural_network/synapse.cpp include/neural_networks/neuron.h src/neural_network/neuron.cpp include/neural_networks/neural_network.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp  include/neural_networks/utils.h src/neural_network/utils.cpp include/neural_networks/message.h src/neural_network/message.cpp include/neural_networks/networks/adaptive_network.h src/neural_network/networks/adaptive_network.cpp include/neural_networks/dynamic_elem.h src/neural_network/dynamic_elem.cpp src/environments/tmaze.cpp include/environments/tmaze.h run_tmaze.cpp )

#Cedar
target_link_libraries(FlexibleNN PRIVATE mariadb)

#GCloud
#target_link_libraries(FlexibleNN PUBLIC tbb)
#target_link_libraries(FlexibleNN PUBLIC mariadb)

project(FlexibleNN LANGUAGES CXX)
