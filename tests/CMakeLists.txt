cmake_minimum_required(VERSION 3.14)
project(TestCases LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-O3 -lpthread")

add_executable(tests test.cpp include/recurrence_tests.h src/recurrent_gradient_testcase.cpp ../src/utils.cpp ../include/utils.h ../include/neural_networks/synapse.h ../src/neural_network/synapse.cpp ../include/neural_networks/neuron.h ../src/neural_network/neuron.cpp ../include/neural_networks/neural_network.h ../include/neural_networks/utils.h ../src/neural_network/utils.cpp ../include/neural_networks/message.h ../src/neural_network/message.cpp ../include/neural_networks/networks/adaptive_network.h include/fixed_recurrent_network.h ../src/neural_network/networks/adaptive_network.cpp src/fixed_recurrent_network.cpp ../include/animal_learning/tracecondioning.h ../src/animal_learning/traceconditioning.cpp src/feedforward_gradient_testcase.cpp include/fixed_feedforward_network.h src/fixed_feedforward_network.cpp ../src/animal_learning/tracepatterning.cpp  ../include/neural_networks/dynamic_elem.h ../src/neural_network/dynamic_elem.cpp ../include/neural_networks/networks/adaptive_recurrent_network.h ../src/neural_network/networks/adaptive_recurrent_networks.cpp src/catch3/catch_amalgamated.cpp include/catch3/catch_amalgamated.hpp ../include/neural_networks/networks/network.h ../src/neural_network/networks/network.cpp)
# For code coverage
if(CMAKE_COMPILER_IS_GNUCXX)
    LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
    set(COVERAGE_LCOV_EXCLUDES 'catch_amalgamated.cpp' 'catch_amalgamated.h')
    SETUP_TARGET_FOR_COVERAGE_LCOV(NAME tests_coverage
            EXECUTABLE tests
            DEPENDENCIES tests)
    endif()
add_test(test1 ./tests)
target_link_libraries(tests PUBLIC tbb gcov)

project(TestCases LANGUAGES CXX)

