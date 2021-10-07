//
// Created by Khurram Javed on 2021-09-22.
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
#include "include/nn/networks/layerwise_feedworward.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"
#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"

int main(int argc, char *argv[]){


  float running_error = 6;
  float accuracy = 0.1;
  Experiment my_experiment = Experiment(argc, argv);

  Metric error_metric = Metric(my_experiment.database_name, "error_table",
                               std::vector < std::string > {"step", "run", "error", "accuracy"},
                               std::vector < std::string > {"int", "int", "real", "real"},
                               std::vector < std::string > {"step", "run"});

  LayerwiseFeedforward network = LayerwiseFeedforward(my_experiment.get_float_param("meta_step_size"), my_experiment.get_int_param("seed"), 28*28, 10, 0.001);

  std::vector<std::vector<std::string>> error_logger;


  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");
  std::mt19937 mt(my_experiment.get_int_param("seed"));
  int total_data_points = 6000;
  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);
//
  mnist::binarize_dataset(dataset);
  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;
  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.training_labels[counter])));
    images.push_back(x_temp);
    targets.push_back(y_temp);
  }

  for (int i = 0; i < my_experiment.get_int_param("steps"); i++) {

    int index = index_sampler(mt);
    auto x = images[index];
    float y_index = targets[index][0];
    std::vector<float> y(10);
    y[y_index] = 1;
//    print_vector(y_gt);
//    exit(1);
//    for(int i = 0; i< 28*28; i++){
//      std::cout << x[i];
//      if(i%28==27)
//        std::cout << std::endl;
//    }
//    std::cout << "\nTarget\t" << y[0] << std::endl;
//    std::cout << x.size() << " " << y.size() << std::endl;
//    continue;
    network.forward(x);
    auto prediction = network.read_output_values();
    float error = 0;
    for(int i = 0; i<prediction.size(); i++){
      error += (prediction[i]-y[i])*(prediction[i]-y[i]);
    }
    running_error = running_error * 0.999 + 0.001 * sqrt(error);
    if(argmax(prediction) == y_index){
      accuracy = accuracy*0.999 + 0.001;
    }
    else{
      accuracy*= 0.999;
    }
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
      error.push_back(std::to_string(accuracy));
      error_logger.push_back(error);

    }
    if(i % 10000 == 0){
      std::cout << error_logger.size() << std::endl;
      error_metric.add_values(error_logger);
      error_logger.clear();
    }

    if (i % 1000 == 0) {
      std::cout << "Step " << i << std::endl;

      std::cout << "Network confing\n";
      std::cout << "No\tSize\tSynapses\tOutput\n";
      for(int layer_no = 0; layer_no < network.LTU_neuron_layers.size(); layer_no++){
        std::cout <<  layer_no << "\t" << network.LTU_neuron_layers[layer_no].size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;

//        std::cout << "Layer sz\t" << network.LTU_neuron_layers[layer_no].size() << std::endl;
//      }
//      for(auto it : network.LTU_neuron_layers){
//        std::cout << "ID\tVal\n";
//        for(auto n : it){
//          std::cout << n->id << "\t" << n->value << std::endl;
//        }
      }


      std::cout << "Running accuracy = " << accuracy << std::endl;
      std::cout << "GT " << y_index <<  " Pred = " << argmax(prediction) << std::endl;
      std::cout << " Target\n";
      print_vector(y);

      std::cout << " Prediction\n";
      print_vector(prediction);
//      std::cout << "Current index " << env.get_index() << std::endl;
//      print_vector(x);
//      std::cout << "Target\t";
//      print_vector(y);
//      std::cout << "Output val\t";
//      print_vector(network.read_output_values());

      std::cout << "Running error = " << running_error << std::endl;
    }
//    if (argmax(prediction) != y_index) {
    if(error > 0.05){
//      if(network.all_synapses.size() < 10000)
//      for(int temp = 0; temp<10; temp ++)
      if(my_experiment.get_int_param("imprint") == 1)
        network.imprint_feature(i, x);
      else
        network.imprint_feature_random();

    }
//    std::cout << "ID\tUtil\n";
//    for(auto n : network.all_neurons){
//      if(!n->is_input_neuron && !n->is_output_neuron){
//        std::cout << n->id << "\t" << n->neuron_utility << std::endl;
//      }
//    }


  }
  error_metric.add_values(error_logger);
  error_logger.clear();
}

