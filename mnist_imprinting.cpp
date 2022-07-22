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
  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector < std::string > {"step", "run", "error", "accuracy"},
                               std::vector < std::string > {"int", "int", "real", "real"},
                               std::vector < std::string > {"step", "run"});
  Metric error_metric_test = Metric(my_experiment->database_name, "test_set",
                               std::vector < std::string > {"step", "run", "accuracy", "mode"},
                               std::vector < std::string > {"int", "int", "real", "int"},
                               std::vector < std::string > {"step", "run", "mode"});

  LayerwiseFeedforward network = LayerwiseFeedforward(my_experiment->get_float_param("step_size"), my_experiment->get_float_param("meta_step_size"), my_experiment->get_int_param("seed"), 28*28, 10, 0.001);

  std::vector<std::vector<std::string>> error_logger;
  std::vector<std::vector<std::string>> error_logger_test;


  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");

  std::mt19937 mt(my_experiment->get_int_param("seed"));
  int total_data_points = my_experiment->get_int_param("training_points");
  int total_test_points = 10000;
  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);
//
  mnist::binarize_dataset(dataset);
  bool training_phase = true;
  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;
//
  std::vector<std::vector<float>> images_test;
  std::vector<std::vector<float>> targets_test;

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

  for(int counter = 0; counter < 10000; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.test_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.test_labels[counter])));
    images_test.push_back(x_temp);
    targets_test.push_back(y_temp);
  }
  int total_steps = 0;
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
    total_steps++;
    int index = index_sampler(mt);
    auto x = images[index];
    float y_index = targets[index][0];
    std::vector<float> y(10);
    y[y_index] = 1;

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
    network.backward(y, training_phase);
    if (i % 100 == 0) {
      std::vector<std::string> error;
      error.push_back(std::to_string(i));
      error.push_back(std::to_string(my_experiment->get_int_param("run")));
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

      }


      std::cout << "Running accuracy = " << accuracy << std::endl;
      std::cout << "GT " << y_index <<  " Pred = " << argmax(prediction) << std::endl;
      std::cout << " Target\n";
      print_vector(y);

      std::cout << " Prediction\n";
      print_vector(prediction);
      std::cout << "Running error = " << running_error << std::endl;
    }
//
    if (argmax(prediction) != y_index && training_phase
//    && (((i%100000) > 50000))
    ) {
//    if(error > 0.01 && i < 1000){
//      if(network.all_synapses.size() < 10000)
//      for(int temp = 0; temp<10; temp ++)

      if(my_experiment->get_int_param("imprint") == 1)
//        for(int temp = 0; temp<10; temp ++)
          network.imprint_feature(i, x, my_experiment->get_float_param("step_size"), my_experiment->get_float_param("meta_step_size"), y_index);
      else
        network.imprint_feature_random(my_experiment->get_float_param("step_size"), my_experiment->get_float_param("meta_step_size"));

    }


    if(i%1000 == 999){
      int correct = 0;
      for(int index = 0; index<total_data_points; index++){
        auto x = images[index];
        float y_index = targets[index][0];
        std::vector<float> y(10);
        y[y_index] = 1;
        network.forward(x);
        auto prediction = network.read_output_values();
        if(argmax(prediction) == y_index){
          correct++;
        }
      }
      std::vector<std::string> error;
      error.push_back(std::to_string(i));
      error.push_back(std::to_string(my_experiment->get_int_param("run")));
      error.push_back(std::to_string(float(correct)/total_data_points));
      error.push_back(std::to_string(0));
      error_logger_test.push_back(error);

      error_metric_test.add_values(error_logger_test);
      error_logger_test.clear();
      std::cout << "Step: " << i <<"\tTrain Accuracy: " << float(correct)/total_data_points << std::endl;
    }


    if(i%10000 == 9999){
      int correct = 0;
      for(int index = 0; index<total_test_points; index++){
        auto x = images_test[index];
        float y_index = targets_test[index][0];
        std::vector<float> y(10);
        y[y_index] = 1;
        network.forward(x);
        auto prediction = network.read_output_values();
        if(argmax(prediction) == y_index){
          correct++;
        }
      }
      std::vector<std::string> error;
      error.push_back(std::to_string(i));
      error.push_back(std::to_string(my_experiment->get_int_param("run")));
      error.push_back(std::to_string(float(correct)/10000));
      error.push_back(std::to_string(1));
      error_logger_test.push_back(error);

      error_metric_test.add_values(error_logger_test);
      error_logger_test.clear();
      std::cout << "Step: " << i <<"\tTest Accuracy: " << float(correct)/10000 << std::endl;
    }



  }

  total_steps++;
  int correct = 0;
  for(int index = 0; index<total_data_points; index++){
    auto x = images[index];
    float y_index = targets[index][0];
    std::vector<float> y(10);
    y[y_index] = 1;
    network.forward(x);
    auto prediction = network.read_output_values();
    if(argmax(prediction) == y_index){
      correct++;
    }
  }
  std::vector<std::string> error;
  error.push_back(std::to_string(total_steps));
  error.push_back(std::to_string(my_experiment->get_int_param("run")));
  error.push_back(std::to_string(float(correct)/total_data_points));
  error.push_back(std::to_string(0));
  error_logger_test.push_back(error);

  error_metric_test.add_values(error_logger_test);
  error_logger_test.clear();
  std::cout << "Step: " << total_steps <<"\tTrain Accuracy: " << float(correct)/total_data_points << std::endl;


  error_metric.add_values(error_logger);
  error_logger.clear();
}

