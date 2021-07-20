//
// Created by Khurram Javed on 2021-07-19.
//


#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include "catch_amalgamated.hpp"
#include "recurrence_tests.h"

TEST_CASE("Recurrent gradient", "[Gradient estimation]") {
    REQUIRE(recurrent_network_test());
}
//}