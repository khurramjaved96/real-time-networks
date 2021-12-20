//
// Created by Khurram Javed on 2021-07-19.
//


#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include <iostream>

#include "include/catch3/catch_amalgamated.hpp"
#include "include/test_cases.h"

//TEST_CASE("Recurrent gradient", "[Gradient estimation]") {
//  REQUIRE(recurrent_network_test());
//}

TEST_CASE("Feedforward deep gradients with variable length paths", "[Gradient estimation relu]") {
  REQUIRE(feedforwadtest_relu());
}

TEST_CASE("Feedforward deep gradients with variable length paths Sigmoid Activation", "[Gradient estimation sigmoid]") {
  REQUIRE(feedforwadtest_sigmoid());
}

TEST_CASE("Feedforward deep gradients with variable length paths LeakyReLU Activation", "[Gradient estimation leaky relu]") {
  REQUIRE(feedforwardtest_leaky_relu());
}

TEST_CASE("Target value test", "[Gradient_estimation_target_network]") {
  REQUIRE(forward_pass_without_sideeffects_test());
}

TEST_CASE("Gradient estimation with mixed activations", "[Gradient_estimation_mixed]") {
  REQUIRE(feedforward_mixed_activations());
}

TEST_CASE("Gradient estimation relu with random inputs", "[Gradient_estimation_random]") {
  REQUIRE(feedforwadtest_relu_random_inputs());
}

TEST_CASE("Forward view Lambda returns gradients", "[Gradient_estimation_lambda]") {
  REQUIRE(lambda_return_test());
}

TEST_CASE("Step-size adaptation test case (IDBD)", "[Learning]") {
  REQUIRE(train_single_parameter());
}

TEST_CASE("Step-size adaptation multiple layers test case", "[Learning]") {
  REQUIRE(train_single_parameter_two_layers());
}

TEST_CASE("TIDBD Test on the simple environment used in the TD(lambda) report", "[Learning]") {
  REQUIRE(train_single_parameter_tidbd_correction_test());
}
//
TEST_CASE("TIDBD test with dead end", "[Learning]") {
  REQUIRE(train_single_parameter_with_no_grad_synapse());
}

TEST_CASE("Mountain car energy pumping on-policy prediction learning test", "[Learning]") {
  REQUIRE(mountain_car_test());
}

//TEST_CASE("Sarsa(lambda) with linear function approximation test", "[SarsaControl]") {
//  REQUIRE(sarsa_lfa_mc_test());
//}

TEST_CASE("LFA test for utility propagation", "[Utility_propagation]") {
  REQUIRE(utility_test());
}

TEST_CASE("Layerwise gradient estimation", "[Layerwise]") {
  REQUIRE(layerwise_seqeuntial_gradient_testcase());
}

  TEST_CASE("Sarsa Lambda test", "[SarsaLambda]") {
    REQUIRE(sarsa_lambda_test());
}
//}