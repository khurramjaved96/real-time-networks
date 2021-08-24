//
// Created by Khurram Javed on 2021-07-22.
//

#ifndef TESTS_INCLUDE_FEEDFORWARD_TESTCASE_H
#define TESTS_INCLUDE_FEEDFORWARD_TESTCASE_H

bool feedforwadtest_relu();

bool feedforwadtest_relu_random_inputs();

bool feedforwadtest_sigmoid();

bool feedforwardtest_leaky_relu();

bool recurrent_network_test();

bool feedforward_relu_with_targets_test();

bool forward_pass_without_sideeffects_test();

bool lambda_return_test();

bool train_single_parameter();

bool train_single_parameter_tidbd_correction_test();

bool train_single_parameter_two_layers();

bool train_single_parameter_with_no_grad_synapse();

bool feedforward_mixed_activations();

bool mountain_car_test();

bool sarsa_lfa_mc_test();

bool utility_test();
#endif  // TESTS_INCLUDE_FEEDFORWARD_TESTCASE_H
