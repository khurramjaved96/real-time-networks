//
// Created by Khurram Javed on 2021-04-25.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include "../include/test_cases.h"

#include <iostream>
#include <vector>

#include "../include/test_case_networks.h"
#include "../../include/nn/networks/network.h"
#include "../../include/nn/neuron.h"
#include "../../include/nn/synapse.h"
#include "../../include/nn/synced_neuron.h"
#include "../../include/nn/synced_synapse.h"
#include "../include/random_data_generator.h"
#include "../../include/utils.h"
#include "../../include/environments/animal_learning/tracecondioning.h"

bool layerwise_seqeuntial_gradient_testcase() {

  auto network = LayerwiseFeedforward(1e-4, 0, 3, 1, 0.001);
  auto four = new ReluSyncedNeuron(false, false);
  four->set_layer_number(1);
  auto five = new ReluSyncedNeuron(false, false);
  five->set_layer_number(1);
  auto six = new ReluSyncedNeuron(false, false);
  six->set_layer_number(2);
  auto seven = new ReluSyncedNeuron(false, false);
  seven->set_layer_number(3);
  auto one = network.input_neurons[0];
  auto two = network.input_neurons[1];
  auto three = network.input_neurons[2];
  auto eight = network.output_neurons[0];

  network.all_neurons.push_back(four);
  network.all_neurons.push_back(five);
  network.all_neurons.push_back(six);
  network.all_neurons.push_back(seven);
  network.LTU_neuron_layers[1].push_back(four);
  network.LTU_neuron_layers[1].push_back(five);
  network.LTU_neuron_layers[2].push_back(six);
  network.LTU_neuron_layers[3].push_back(seven);

  network.all_synapses.push_back(new SyncedSynapse(one, four, 0.02, 0));
  network.all_synapses.push_back(new SyncedSynapse(two, six, -0.1, 0));
  network.all_synapses.push_back(new SyncedSynapse(four, six, 0.06, 0));
  network.all_synapses.push_back(new SyncedSynapse(two, five, 0.4, 0));
  network.all_synapses.push_back(new SyncedSynapse(three, five, 0.2, 0));
  network.all_synapses.push_back(new SyncedSynapse(five, seven, 0.1, 0));
  network.all_synapses.push_back(new SyncedSynapse(six, seven, 0.2, 0));

  auto s = new SyncedSynapse(six, eight, .09, 0);
  network.all_synapses.push_back(s);
  network.output_synapses.push_back(s);

  s = new SyncedSynapse(seven, eight, 0.2, 0);
  network.all_synapses.push_back(s);
  network.output_synapses.push_back(s);


  for(int steps = 0; steps < 1000; steps++) {
    std::vector<float> inp;
    inp.push_back(0.6);
    inp.push_back(1.2);
    inp.push_back(2.5);

    std::vector<float> target;
    target.push_back(0.2);
//  target.push_back(0.8);

//    std::cout << "Forward pass\n";
    network.forward(inp);
//    std::cout << "Backward pass\n";
    network.backward(target);
//    std::cout << "Backward pass done\n";
  }

  std::cout << "ID\tVal\tVal Before Fire\tUtil\n";
  for(auto n : network.all_neurons){
    std::cout << n->id <<"\t" << n->value << "\t" << n->value_before_firing << "\t\t" << n->neuron_utility << std::endl;
  }
  std::cout << "\n\nID\tW\tG\tUtil\n";
  for(auto n : network.all_synapses){
    std::cout << n->id << "\t" << n->weight << "\t" << n->credit << "\t" << n->synapse_utility << std::endl;
  }


  return true;
}

bool feedforward_relu_with_targets_test() {
  TestCase my_network = TestCase(0.0, 5, 5);
  long long int time_step = 0;
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
  float running_error = -1;

  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 300; a++) {
    std::vector<float> curr_inp;

    if (a < 100) {
      curr_inp.push_back(-10000);
      curr_inp.push_back(-10000);
      curr_inp.push_back(-10000);
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
    if (counter < 200 and counter >= 100) {
      output[0]--;
    }
//    if (counter < 199 && counter > 100) {
//      sum_of_activation += output2[5];
////            std::cout << output2[5] << std::endl;
//    }

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

bool train_single_parameter_with_no_grad_synapse() {
  IDBDLearningNetwork my_network = IDBDLearningNetwork();
  auto input_neuron = new LinearNeuron(true, false);
  auto output_neuron = new LinearNeuron(false, true);
  auto dead_end_neuron = new LinearNeuron(false, false);

  my_network.all_neurons.push_back(dead_end_neuron);

  my_network.input_neurons.push_back(input_neuron);
  my_network.all_neurons.push_back(input_neuron);
  my_network.output_neurons.push_back(output_neuron);
  my_network.all_neurons.push_back(output_neuron);

  auto middle_neuron = new LinearNeuron(false, false);
  middle_neuron->drinking_age = 100000000;
  my_network.all_neurons.push_back(middle_neuron);

  auto my_synapse = new synapse(input_neuron, middle_neuron, 0, 1e-10);
  auto my_synapse_out = new synapse(middle_neuron, output_neuron, 1, 0);
  auto my_synapse_dead_end = new synapse(middle_neuron, dead_end_neuron, 1, 0);
  my_network.all_synapses.push_back(my_synapse);
  my_network.all_synapses.push_back(my_synapse_out);
  my_network.all_synapses.push_back(my_synapse_dead_end);
  my_synapse->turn_on_idbd();
  my_synapse->set_meta_step_size(1e-2);

  std::vector<float> inp;
  inp.push_back(1);
//
  std::vector<float> target;
  target.push_back(5);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::mt19937 mt;
  mt.seed(0);

  for (int step = 0; step < 10000000; step++) {
    target[0] = 50 + dist(mt);
    my_network.set_input_values(inp);
    my_network.step();
    my_network.introduce_targets(target, 0, 0);
//    std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
    if (std::abs(my_synapse->weight - 50) < 0.0001) {
      return true;
    }
  }
  std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
  return false;
}

bool train_single_parameter() {
  IDBDLearningNetwork my_network = IDBDLearningNetwork();
  auto input_neuron = new LinearNeuron(true, false);
  auto output_neuron = new LinearNeuron(false, true);

  my_network.input_neurons.push_back(input_neuron);
  my_network.all_neurons.push_back(input_neuron);
  my_network.output_neurons.push_back(output_neuron);
  my_network.all_neurons.push_back(output_neuron);

  auto my_synapse = new synapse(input_neuron, output_neuron, 0, 1e-10);
  my_network.all_synapses.push_back(my_synapse);
  my_synapse->turn_on_idbd();
  my_synapse->set_meta_step_size(1e-2);

  std::vector<float> inp;
  inp.push_back(1);

  std::vector<float> target;
  target.push_back(10);

  std::uniform_real_distribution<float> dist(-0.1, 0.1);
  std::mt19937 mt;
  mt.seed(0);

  for (int step = 0; step < 10000; step++) {
    target[0] = 50 + dist(mt);
    my_network.set_input_values(inp);
    my_network.step();
    my_network.introduce_targets(target, 0, 0);
//    std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
    if (std::abs(my_synapse->weight - 50) < 0.001) {
      return true;
    }
  }

  std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
  return false;
}

bool train_single_parameter_tidbd_correction_test() {
  IDBDLearningNetwork my_network = IDBDLearningNetwork();
  auto input_neuron = new LinearNeuron(true, false);
  auto output_neuron = new LinearNeuron(false, true);

  my_network.input_neurons.push_back(input_neuron);
  my_network.all_neurons.push_back(input_neuron);
  my_network.output_neurons.push_back(output_neuron);
  my_network.all_neurons.push_back(output_neuron);

  auto my_synapse = new synapse(input_neuron, output_neuron, 0, 1e-10);
  my_network.all_synapses.push_back(my_synapse);
  my_synapse->turn_on_idbd();
  my_synapse->set_meta_step_size(1e-2);

  std::vector<float> inp;
  inp.push_back(1);

  std::vector<float> target;
  target.push_back(10);

  std::uniform_real_distribution<float> dist(-0.1, 0.1);
  std::mt19937 mt;
  mt.seed(0);
  int state = 0;
  for (int step = 0; step < 100000; step++) {
    int feature;
    int target_c;
    int lambda = 1;
    int gamma = 1;
    if (state == 0) {
      feature = 1;
      target_c = 0;
    } else if (state == 1) {
      feature = 0;
      target_c = 1;
    } else {
      feature = 0;
      target_c = 0;
      gamma = 0;
    }
    inp[0] = feature;

    target[0] = target_c;
    my_network.set_input_values(inp);
    my_network.step();
    my_network.introduce_targets(target, gamma, lambda);

    state = (state + 1) % 3;
//    std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
    if (std::abs(my_synapse->weight - 1) < 0.001) {
      return true;
    }
  }

  std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
  return false;
}

bool train_single_parameter_two_layers() {
  IDBDLearningNetwork my_network = IDBDLearningNetwork();
  auto input_neuron = new LinearNeuron(true, false);
  auto output_neuron = new LinearNeuron(false, true);

  my_network.input_neurons.push_back(input_neuron);
  my_network.all_neurons.push_back(input_neuron);
  my_network.output_neurons.push_back(output_neuron);
  my_network.all_neurons.push_back(output_neuron);

  auto middle_neuron = new LinearNeuron(false, false);
  middle_neuron->drinking_age = 100000000;
  my_network.all_neurons.push_back(middle_neuron);

  auto my_synapse = new synapse(input_neuron, middle_neuron, 0, 1e-10);
  auto my_synapse_out = new synapse(middle_neuron, output_neuron, 1, 0);
  my_network.all_synapses.push_back(my_synapse);
  my_network.all_synapses.push_back(my_synapse_out);
  my_synapse->turn_on_idbd();
  my_synapse->set_meta_step_size(1e-2);

  std::vector<float> inp;
  inp.push_back(1);
//
  std::vector<float> target;
  target.push_back(5);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::mt19937 mt;
  mt.seed(0);

  for (int step = 0; step < 10000000; step++) {
    target[0] = 50 + dist(mt);
    my_network.set_input_values(inp);
    my_network.step();
    my_network.introduce_targets(target, 0, 0);
//    std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
    if (std::abs(my_synapse->weight - 50) < 0.0001) {
      return true;
    }
  }
  std::cout << "Weight " << my_synapse->weight << " Step-size " << my_synapse->step_size << std::endl;
  return false;
}

bool lambda_return_test() {
  RandomDataGenerator env = RandomDataGenerator();
  LambdaReturnNetwork my_network = LambdaReturnNetwork();
  bool done = false;
  int counter = 0;
  while (!done) {
    std::vector<float> inp = env.get_input();
    std::vector<float> target = env.get_target();
    done = env.step();
    my_network.set_input_values(inp);
    my_network.step();
    auto temp_targets = my_network.read_output_values();

    if (counter <= 99) {
      my_network.introduce_targets(target, 0.98, 0.94);
    } else {
      my_network.introduce_targets(temp_targets, 0.98, 0.94);
    }
    counter++;
  }
//  print_vector(my_network.sum_of_gradients);
  std::vector<float> gt
      {-129352.5859, 668060.1250, 741512.1875, -446644.2188, 1248485.1250,
       1298128.0000, 1425246.7500};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1) {
      std::cout << "Test failed \n";
      std::cout << "COMPUTED GRADIENT\n";
      print_vector(my_network.sum_of_gradients);
      std::cout << "GT GRADIENT\n";
      print_vector(gt);
      return false;
    }
    counter_gt++;
  }
  return true;
}
bool feedforwadtest_relu_random_inputs() {
//
  TestCase my_network = TestCase(0.0, 5, 5);
  long long int time_step = 0;
  float running_error = -1;

  std::vector<float>
      l1{39, -59, 0, -52, -12, -83, -6, 62, 7, -15, -98, -19, 85, -46, 74, 27, 31, -17, 54, 74, 90, -30, -77, 29, 90,
         -84, 25, -50, 81, 18, -81, -60, 76, -97, 58, -43, -73, -1, -100, 37, 0, -15, -62, -27, -38, -86, -38, 82, -58,
         95, 50, 81, 68, 69, 19, -42, 90, -21, 62, -66, -53, -83, 79, -17, -23, -95, -40, -21, -59, 75, -91, 7, 52, 26,
         91, -19, 50, -15, 62, 62, 39, -30, 18, 39, 54, 82, -19, 27, -68, 65, -73, -46, 17, 53, 66, -59, 54, 34, -47,
         -8};

  std::vector<float>
      l2{34, -61, 75, -86, -98, -55, 31, -74, 21, -48, -87, 56, -72, 93, 10, 67, -21, -3, 79, -36, -29, 69, -30, 49,
         84, 97, -40, -78, 51, 57, -14, 23, -51, 96, 23, 42, -55, 72, -42, 22, 54, -45, 72, -24, -58, 15, 99, -84, -77,
         -68, 67, -50, -75, -40, 96, 99, -59, 80, 93, -75, -67, 83, 47, 6, -39, -92, 44, -58, 67, -63, -12, -77, -8,
         59, -56, -66, -28, 14, 35, -88, 43, 25, -8, 78, 44, 31, 76, -80, 94, -46, -49, -75, 55, -36, -12, -91, 79,
         -79, 17, 78};

  std::vector<float>
      l3{-65, -18, 63, 25, 61, 97, -17, 26, 79, -79, -22, 77, 6, 35, -85, 1, 52, 12, -18, 56, 10, 25, 76, 85, 42, -54,
         -13, 34, -64, 16, -26, 42, -52, -45, 80, -88, 34, -57, -49, -70, -15, -81, 16, -18, -82, 3, -71, -31, 39, -68,
         42, 100, 86, -24, 86, -21, 0, 92, -86, 36, -68, 55, -10, -88, -41, -14, -61, -5, 11, -89, 16, -70, 80, 85,
         -70, 72, -4, 87, 83, 6, -75, 69, 65, 71, -72, -73, -21, 61, -68, 61, -55, -41, 39, 14, 34, -3, 71, 11, 15,
         -58};
  std::vector<std::vector<float>> input_list;
  for (int a = 0; a < 300; a++) {
    std::vector<float> curr_inp;

    if (a < 100) {
      curr_inp.push_back(l1[a]);
      curr_inp.push_back(l2[a]);
      curr_inp.push_back(l3[a]);
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

//    std::cout << "Sum of activation = " << sum_of_activation << std::endl;
  std::vector<float> gt
      {-91.20002746582031, 517.199951171875, 78.239990234375, 539.7999267578125, -205.04000854492188, 79.80000305175781,
       235.0, 248.1499786376953, 1822.900146484375, 172.52000427246094, 1756.6802978515625, 1541.1500244140625};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
      std::cout << "Test failed \n";
      std::cout << "COMPUTED GRADIENT\n";
      print_vector(my_network.sum_of_gradients);
      std::cout << "GT GRADIENT\n";
      print_vector(gt);
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

bool feedforward_mixed_activations() {
//
  MixedActivationTest my_network = MixedActivationTest();
  long long int time_step = 0;
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


//    std::cout << "Sum of activation = " << sum_of_activation << std::endl;
  std::vector<float> gt
      {-3.979248046875, 9.690000534057617, -47.171512603759766, 93.51539611816406, -397.9248046875, 530.0460205078125,
       4950.0, 600.4005126953125, 3064.661376953125, 318.1327819824219, 274.1940002441406, 624.625};
  int counter_gt = 0;
  for (auto it: gt) {
    if (std::abs(it - my_network.sum_of_gradients[counter_gt]) > 1e-3) {
      std::cout << "Test failed \n";
      std::cout << "COMPUTED GRADIENT\n";
      print_vector(my_network.sum_of_gradients);
      std::cout << "GT GRADIENT\n";
      print_vector(gt);
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