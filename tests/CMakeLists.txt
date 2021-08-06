cmake_minimum_required(VERSION 3.14)
project(TestCases LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-O3 -lpthread")

add_executable(tests test.cpp ../src/utils.cpp ../include/utils.h ../include/nn/synapse.h ../src/nn/synapse.cpp
        ../include/nn/neuron.h ../src/nn/neuron.cpp ../include/nn/utils.h ../src/nn/utils.cpp ../include/nn/message.h
        ../src/nn/message.cpp ../include/nn/networks/feedforward_state_value_network.h include/fixed_recurrent_network.h
        ../src/nn/networks/feedforward_state_value_network.cpp ../include/environments/animal_learning/tracecondioning.h
        ../src/environments/animal_learning/traceconditioning.cpp src/feedforward_gradient_testcase.cpp src/target_without_sideffects.cpp
        include/fixed_feedforward_network.h src/network_implementations.cpp
        ../src/environments/animal_learning/tracepatterning.cpp ../include/nn/dynamic_elem.h ../src/nn/dynamic_elem.cpp
        ../include/nn/networks/recurrent_state_value_network.h  src/catch3/catch_amalgamated.cpp
        include/catch3/catch_amalgamated.hpp ../include/nn/networks/network.h ../src/nn/networks/network.cpp)
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

