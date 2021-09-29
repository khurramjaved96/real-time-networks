//
// Created by Khurram Javed on 2021-09-17.
//

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

//
// Created by Khurram Javed on 2021-04-01.
//


#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <random>
#include <cmath>

#include "include/utils.h"
#include "include/environments/supervised_imprinting.h"
#include "include/nn/networks/imprinting_supervised_network.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"

/**
 * Our main entry function for running all experiments.
 * @param argc Number of arguments
 * @param argv This needs to include the following parameters:
 *  --run (int, 0), the run number
 *  --ISI_low (int, 14), the ISI is sampled based on a uniform distribution. What is the lower bound for this distribution?
 *  --ISI_high (int, 26), what is the upper bound for this distribution?
 *  --lambda (float, 0.0), parameter for an eligibility trace. What is our trace parameter?
 *  --seed (int, 2021), what is the seed we use?
 *  --width (int, 6), [NOT CURRENTLY USED] what is the width of our neural network?
 *  --step_size (float, 0.0001), step size parameter.
 *  --steps (int, 5000000), total number of steps to take in the experiment.
 * @return void
 */
int main(int argc, char *argv[]) {
  float running_error = 6;
  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"step", "run", "error"},
                               std::vector < std::string > {"int", "int", "real"},
                               std::vector < std::string > {"step", "run"});


  SupervisedImprintingEnv env = SupervisedImprintingEnv(my_experiment.get_int_param("seed")*2);
  ImprintingSupervised network = ImprintingSupervised(my_experiment.get_float_param("meta_step_size"), my_experiment.get_int_param("seed"), 30, 0.001, my_experiment.get_int_param("features"));

  std::vector<std::vector<std::string>> error_logger;


  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {

    auto x = env.get_x();
    network.forward(x);
    auto target = network.read_output_values();
    auto y = env.get_y();
    float error = (target[0] - y[0]) * (target[0] - y[0]);
    running_error = running_error * 0.9999 + 0.0001 * sqrt(error);
//    std::cout << "Error = " << error << std::endl;
//    print_vector(target);
//    print_vector(y);
//    exit(1);
    network.backward(y);
    if (i % 100 == 0) {
      std::vector<std::string> error;
      error.push_back(std::to_string(i));
      error.push_back(std::to_string(my_experiment.get_int_param("run")));
      error.push_back(std::to_string(running_error));
      error_logger.push_back(error);

    }
    if(i % 100000 == 0){
      std::cout << error_logger.size() << std::endl;
      error_metric.add_values(error_logger);
      error_logger.clear();
    }

    if (i % 1000 == 0) {
      std::cout << "Step " << i << std::endl;
//      std::cout << "Current index " << env.get_index() << std::endl;
//      print_vector(x);
//      std::cout << "Target\t";
//      print_vector(y);
//      std::cout << "Output val\t";
//      print_vector(network.read_output_values());
      std::cout << "Running error = " << running_error << std::endl;
    }
    if (error > 1) {
      if(my_experiment.get_int_param("imprinting") == 0)
        network.imprint_feature(i, x);
      else if(my_experiment.get_int_param("imprinting") == 1)
        network.imprint_feature(i, env.create_pattern());
    }
    env.step();

  }
  return 0;
}

