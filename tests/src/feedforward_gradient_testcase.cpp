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

bool feedforwadtest_sigmoid() {
//
  TestCase my_network = TestCase(0.0, 5, 5, true);
  long long int time_step = 0;
  std::cout << "Total synapses in the network " << my_network.get_total_synapses() << std::endl;
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 300; a++) {
    std::vector<float> curr_inp;
    if (a < 100) {
      curr_inp.push_back(-1000);
      curr_inp.push_back(-1000);
      curr_inp.push_back(-1000);
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
      output[0]++;
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
      {-0.0042365542612969875, 2.214078664779663, -1.181681752204895, 0.0500468909740448, -0.4236554205417633,
       0.1948026418685913, 4.421022891998291, 94.388671875, 0.0061254040338099, 97.9652099609375, 59.38797378540039};
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
      output[0]++;
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
      {-3.9099998474121094, 9.505999565124512, -45.89999771118164, 93.43999481201172, -391.0, 525.6000366210938, 4851.0,
       587.7637329101562, 3002.000732421875, 315.5230712890625, 274.3168029785156, 611.5283203125};
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
      output[0]++;
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
      {-3.9184300899505615, 9.505999565124512, -45.94569778442383, 93.51499938964844, -391.843017578125,
       529.8500366210938, 4851.0, 587.7557373046875, 3001.9609375, 318.23748779296875, 273.9898681640625,
       611.520263671875};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
      return false;
    }
    counter_gt++;
  }

  return true;

}

bool recurrent_network_test() {
  ContinuallyAdaptingRecurrentNetworkTest my_network = ContinuallyAdaptingRecurrentNetworkTest(0.0, 5, 5);
  std::vector<float> sum_of_gradients;
  for (auto it: my_network.all_synapses)
    sum_of_gradients.push_back(0);

  std::vector<std::vector<float>> input_list;

  for (int a = 0; a < 100; a++) {
    std::vector<float> curr_inp;
    curr_inp.push_back(0);
    curr_inp.push_back(0);
    input_list.push_back(curr_inp);
  }
  for (int a = 0; a < 100; a++) {
    std::vector<float> curr_inp;

    curr_inp.push_back(10 - (a * 0.1));
    curr_inp.push_back(a * 0.01);
    input_list.push_back(curr_inp);
  }
  for (int a = 0; a < 100; a++) {
    std::vector<float> curr_inp;
    curr_inp.push_back(0);
    curr_inp.push_back(0);
    input_list.push_back(curr_inp);
  }

  float sum_of_state = 0;
  int counter = 0;
  for (auto it:input_list) {
    my_network.set_input_values(it);
    my_network.step();
    std::vector<float> output = my_network.read_output_values();

    if (counter >= 100 && counter < 200) {
      sum_of_state += my_network.read_all_values()[2];
      output[0]++;
    }
    my_network.introduce_targets(output, 0, 0);
    int counter_temp = 0;
    for (auto it: my_network.all_synapses) {
      sum_of_gradients[counter_temp] += it->credit;
      counter_temp++;
    }
    counter++;
  }
// Ground truth is computed by running "recurrent_test.py" in python_scripts
  std::vector<float> gt{877.8438, 74.0906, 1824.9984, 1043.9813};
  for (int i = 0; i < sum_of_gradients.size(); i++) {
    if (std::abs(sum_of_gradients[i] - gt[i]) > 1e-3)
      return false;
  }
  return true;
}