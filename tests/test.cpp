//
// Created by Khurram Javed on 2021-07-19.
//


#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch_amalgamated.hpp"
#include "recurrence_tests.h"


//GT : 877.8438, 74.0906, 1824.9984, 1043.9813
TEST_CASE( "Recurrent gradient", "[Gradient estimation]" ) {
REQUIRE( recurrent_network_test());
}