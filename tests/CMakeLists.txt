cmake_minimum_required(VERSION 3.14)
project(TestCases LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_CXX_FLAGS "-O3 -lpthread")

add_executable(tests test.cpp recurrence_tests.cpp ../src/utils.cpp ../include/utils.h ../include/neural_networks/synapse.h ../src/neural_network/synapse.cpp ../include/neural_networks/neuron.h ../src/neural_network/neuron.cpp ../include/neural_networks/neural_network.h ../include/neural_networks/utils.h ../src/neural_network/utils.cpp ../include/neural_networks/message.h ../src/neural_network/message.cpp ../include/neural_networks/networks/adaptive_network.h ../include/neural_networks/networks/test_recurrent.h ../src/neural_network/networks/adaptive_network.cpp ../src/neural_network/networks/test_recurrent.cpp ../include/animal_learning/tracecondioning.h ../src/animal_learning/traceconditioning.cpp ../test_case.cpp ../include/neural_networks/networks/test.h ../src/neural_network/networks/test.cpp ../src/animal_learning/tracepatterning.cpp  ../include/neural_networks/dynamic_elem.h ../src/neural_network/dynamic_elem.cpp ../include/neural_networks/networks/adaptive_recurrent_network.h ../src/neural_network/networks/adaptive_recurrent_networks.cpp  catch_amalgamated.cpp catch_amalgamated.hpp)

# For linking locally
target_link_libraries(tests PUBLIC tbb)
#target_link_libraries(FlexibleNN PUBLIC mariadb)

project(TestCases LANGUAGES CXX)

