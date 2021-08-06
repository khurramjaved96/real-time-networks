//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include "../include/gradient_testcases.h"

#include <iostream>
#include <vector>

#include "../include/fixed_feedforward_network.h"
#include "../include/fixed_recurrent_network.h"
#include "../../include/utils.h"
#include "../../include/environments/animal_learning/tracecondioning.h"

bool feedforward_relu_with_targets_test() {
  TestCase my_network = TestCase(0.0, 5, 5);
  long long int time_step = 0;
  std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  std::vector<std::vector<float>> target_list;
  for (int a = 0; a < 200; a++) {
    std::vector<float> curr_inp;
    std::vector<float> curr_target;

    if (a < 100) {
      curr_inp.push_back((a - 0) * 0.01);
      curr_inp.push_back(10 - ((a - 0) * 0.1));
      curr_inp.push_back(a - 0);
      curr_target.push_back(a * 0.01);
    } else {
      curr_inp.push_back(0);
      curr_inp.push_back(0);
      curr_inp.push_back(0);
      curr_target.push_back(0);
    }
    input_list.push_back(curr_inp);
    target_list.push_back(curr_target);
  }
  int counter = 0;
  float sum_of_activation = 0;
  for (auto it : input_list) {
    my_network.set_input_values(it);
    my_network.step();
    std::vector<float> output = my_network.read_output_values();
    std::vector<float> output2 = my_network.read_all_values();

//        std::cout << "counter = " << counter << std::endl;

//        print_vector(output);
    if (counter < 100) {
      my_network.introduce_targets(target_list[counter]);
      sum_of_activation += output2[5];
    } else {
      my_network.introduce_targets(output);
    }
    counter++;
  }
  std::cout << "Feedforward_relu_with_targets_test\n";
  print_vector(my_network.sum_of_gradients);
//    std::cout << "Sum of activation = " << sum_of_activation << std::endl;
  std::vector<float> gt
      {35.02164840698242, -92.67996978759766, 191.6849822998047, -323.7611083984375, 3502.1630859375, -3540.669921875,
       -47011.26953125, -5867.7744140625, -29793.015625, -2196.6337890625, -706.86474609375, -6099.47216796875};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-1) {
      std::cout << it - my_network.sum_of_gradients[counter_gt] << std::endl;
      return false;
    }
    counter_gt++;
  }

  return true;
}

bool feedforwadtest_sigmoid() {
//
  TestCase my_network = TestCase(0.0, 5, 5, true);
  long long int time_step = 0;
  std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 200; a++) {
    std::vector<float> curr_inp;

    if (a < 100) {
      curr_inp.push_back((a) * 0.01);
      curr_inp.push_back(10 - ((a) * 0.1));
      curr_inp.push_back(a);
    } else {
      curr_inp.push_back(0);
      curr_inp.push_back(0);
      curr_inp.push_back(0);
    }
    input_list.push_back(curr_inp);
  }
  int counter = 0;
  float sum_of_activation = 0;
  for (auto it : input_list) {
    my_network.set_input_values(it);
    my_network.step();
    std::vector<float> output = my_network.read_output_values();
    std::vector<float> output2 = my_network.read_all_values();

//        std::cout << "counter = " << counter << std::endl;

//        print_vector(output);
    if (counter < 100) {
      output[0]--;
    }
    if (counter < 199 && counter > 100) {
      sum_of_activation += output2[5];
//            std::cout << output2[5] << std::endl;
    }

    my_network.introduce_targets(output);
    counter++;
  }
//    std::cout << "Sum of activation "  << sum_of_activation << std::endl;
//    int counter_tt = 0;
//    for (auto it: my_network.all_synapses) {
//        std::cout << it->input_neuron->id + 1 << " " << it->output_neuron->id + 1  << " " << my_network.sum_of_gradients[counter_tt] << std::endl;
//        counter_tt++;
//    }
  print_vector(my_network.sum_of_gradients);

//    std::cout << "Sum of activation = " << sum_of_activation << std::endl;
  std::vector<float> gt
      {-0.0042365542612969875, 2.2576797008514404, -1.181681752204895, 0.05004924535751343, -0.4236554205417633,
       0.19595679640769958, 4.465513706207275, 95.388671875, 0.0061371810734272, 98.96515655517578,
       60.05393981933594};
  print_vector(gt);
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
      return false;
    }
    counter_gt++;
  }

  return true;

}

bool feedforwadtest_relu() {
//
  TestCase my_network = TestCase(0.0, 5, 5);
  long long int time_step = 0;
  std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 300; a++) {
    std::vector<float> curr_inp;

    if (a < 100) {
      curr_inp.push_back((a - 0) * 0.01);
      curr_inp.push_back(10 - ((a - 0) * 0.1));
      curr_inp.push_back(a - 0);
    } else {
      curr_inp.push_back(0);
      curr_inp.push_back(0);
      curr_inp.push_back(0);
    }
    input_list.push_back(curr_inp);
  }
  int counter = 0;
  float sum_of_activation = 0;
  for (auto it : input_list) {
    my_network.set_input_values(it);
    my_network.step();
    std::vector<float> output = my_network.read_output_values();
    std::vector<float> output2 = my_network.read_all_values();

//        std::cout << "counter = " << counter << std::endl;

//        print_vector(output);
    if (counter < 100) {
      output[0]--;
      sum_of_activation += output2[5];
//      std::cout << output2[5] << std::endl;
    }

    my_network.introduce_targets(output);
    counter++;
  }

  print_vector(my_network.sum_of_gradients);
//    std::cout << "Sum of activation = " << sum_of_activation << std::endl;
  std::vector<float> gt
      {-3.9691998958587646, 9.70199966430664, -45.907997131347656, 93.43999481201172, -396.91998291015625,
       525.6000366210938, 4950.0, 600.4005126953125, 3065.856689453125, 315.5230712890625, 274.3168029785156,
       624.655029296875};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
      return false;
    }
    counter_gt++;
  }

  return true;

}

bool feedforwardtest_leaky_relu() {
//
  LeakyReluTest my_network = LeakyReluTest(0.0, 5, 5);
  long long int time_step = 0;
  std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 300; a++) {
    std::vector<float> curr_inp;
    if (a < 100) {
      curr_inp.push_back(0);
      curr_inp.push_back(0);
      curr_inp.push_back(0);
    } else if (a < 200) {
      curr_inp.push_back((a - 100) * 0.01);
      curr_inp.push_back(10 - ((a - 100) * 0.1));
      curr_inp.push_back(a - 100);
    } else {
      curr_inp.push_back(0);
      curr_inp.push_back(0);
      curr_inp.push_back(0);
    }
    input_list.push_back(curr_inp);
  }
  int counter = 0;
  float sum_of_activation = 0;
  for (auto it : input_list) {
    my_network.set_input_values(it);
    my_network.step();
    std::vector<float> output = my_network.read_output_values();
    std::vector<float> output2 = my_network.read_all_values();

//        std::cout << "counter = " << counter << std::endl;

//        print_vector(output);
    if (counter < 200 && counter >= 100) {
      output[0]--;
    }
    if (counter < 199 && counter > 100) {
      sum_of_activation += output2[5];
//            std::cout << output2[5] << std::endl;
    }

    my_network.introduce_targets(output);
    counter++;
  }

  print_vector(my_network.sum_of_gradients);
//    std::cout << "Sum of activation = " << sum_of_activation << std::endl;
  std::vector<float> gt
      {-3.978018045425415, 9.70199966430664, -45.95381546020508, 93.51539611816406, -397.80181884765625,
       530.0460205078125, 4950.0, 600.3925170898438, 3065.81689453125, 318.3638610839844, 273.9629211425781,
       624.64697265625};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
      return false;
    }
    counter_gt++;
  }

  return true;

}

//bool recurrent_network_test() {
//  ContinuallyAdaptingRecurrentNetworkTest my_network = ContinuallyAdaptingRecurrentNetworkTest(0.0, 5, 5);
//  std::vector<float> sum_of_gradients;
//  for (auto it: my_network.all_synapses)
//    sum_of_gradients.push_back(0);
//
//  std::vector<std::vector<float>> input_list;
//
//  for (int a = 0; a < 100; a++) {
//    std::vector<float> curr_inp;
//    curr_inp.push_back(0);
//    curr_inp.push_back(0);
//    input_list.push_back(curr_inp);
//  }
//  for (int a = 0; a < 100; a++) {
//    std::vector<float> curr_inp;
//
//    curr_inp.push_back(10 - (a * 0.1));
//    curr_inp.push_back(a * 0.01);
//    input_list.push_back(curr_inp);
//  }
//  for (int a = 0; a < 100; a++) {
//    std::vector<float> curr_inp;
//    curr_inp.push_back(0);
//    curr_inp.push_back(0);
//    input_list.push_back(curr_inp);
//  }
//
//  float sum_of_state = 0;
//  int counter = 0;
//  for (auto it:input_list) {
//    my_network.set_input_values(it);
//    my_network.step();
//    std::vector<float> output = my_network.read_output_values();
//
//    if (counter >= 100 && counter < 200) {
//      sum_of_state += my_network.read_all_values()[2];
//      output[0]++;
//    }
//    my_network.introduce_targets(output, 0, 0);
//    int counter_temp = 0;
//    for (auto it: my_network.all_synapses) {
//      sum_of_gradients[counter_temp] += it->credit;
//      counter_temp++;
//    }
//    counter++;
//  }
//// Ground truth is computed by running "recurrent_test.py" in python_scripts
//  std::vector<float> gt{877.8438, 74.0906, 1824.9984, 1043.9813};
//  for (int i = 0; i < sum_of_gradients.size(); i++) {
//    if (std::abs(sum_of_gradients[i] - gt[i]) > 1e-3)
//      return false;
//  }
//  return true;
//}