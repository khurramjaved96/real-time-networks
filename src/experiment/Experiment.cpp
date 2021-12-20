//
// Created by Khurram Javed on 2021-03-14.
//

#include "../../include/experiment/Experiment.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include "../../include/experiment/Database.h"

std::map<std::string, std::vector<std::string>> Experiment::parse_params(int total_prams, char **pram_list) {
  bool argument = false;
  std::map<std::string, std::vector<std::string>> result;
  std::string key;
  for (int temp = 0; temp < total_prams; temp++) {
    std::string s(pram_list[temp]);
    if (s.find("--") != std::string::npos) {
//            -- available os a new parameter
      argument = true;
      key = pram_list[temp];
      key = key.substr(2, key.size() - 2);
      std::vector<std::string> my_vec;
      result.insert(std::pair < std::string, std::vector < std::string >> (key, my_vec));
    } else if (argument) {
      std::string value = pram_list[temp];
      result[key].push_back(value);
    }
  }
  return result;
}

std::vector<int> Experiment::frequency_of_params(std::map<std::string, std::vector<std::string>> &args) {
  int total_combinations = 1;
  std::vector<int> size_of_params;
  for (auto it = args.begin();
       it != args.end(); it++) {
    if (it->first == "run") {
      assert(it->second.size() == 1);
    }
    size_of_params.push_back(it->second.size());
    total_combinations *= it->second.size();
  }
  return size_of_params;
}

Experiment::Experiment(int argc, char *argv[]) {

  this->args = this->parse_params(argc, argv);
  std::vector<int> size_of_params = this->frequency_of_params(this->args);
  if (this->args.count("run")) {
    this->run = std::stoi(this->args["run"][0]);
  } else {
    std::cout << "Run number not provided; for example, pass --run 0 as command line argument. Exiting \n";
    exit(0);
  }

  int temp_rank = this->run;

  std::vector<int> selected_combinations;
  for (int &size_of_param : size_of_params) {
    selected_combinations.push_back(temp_rank % size_of_param);
    temp_rank = temp_rank / size_of_param;
  }
  int temp_counter = 0;
  for (auto &arg : this->args) {
    this->args_for_run.insert(
        std::pair<std::string, std::string>(arg.first, arg.second[selected_combinations[temp_counter]]));
    std::cout << arg.first << " " << arg.second[selected_combinations[temp_counter]] << std::endl;
    temp_counter++;
  }

  this->database_name = "khurram_" + this->args_for_run["name"];
  this->d.create_database(this->database_name);
  std::vector<std::string> keys, values, types;
  for (auto const &imap: this->args_for_run) {
    keys.push_back(imap.first);
    if (!imap.second.empty() && imap.second.find_first_not_of("-0123456789") == std::string::npos) {
      types.emplace_back("int");
    } else if (!imap.second.empty() && imap.second.find_first_not_of("-.0123456789") == std::string::npos) {
      types.emplace_back("real");
    } else
      types.emplace_back("text");

    values.push_back(imap.second);

  }
  this->d.make_table(this->database_name, "runs", keys, types, std::vector < std::string > {"run"});
  this->d.add_row_to_table(this->database_name, "runs", keys, values);
}

int Experiment::get_int_param(const std::string &param) {
//    std::cout << "Param count " << param << this->args_for_run.count(param) << " " << std::endl;
  if (this->args_for_run.count(param) == 0) {
    std::cout << "Param does not exist\n";
    throw std::invalid_argument("Param " + param + " does not exist");
  }
  return std::stoi(this->args_for_run[param]);
}

float Experiment::get_float_param(const std::string &param) {
//    std::cout << "Param count " << param << this->args_for_run.count(param) << " " << std::endl;
  if (this->args_for_run.count(param) == 0) {
    std::cout << "Param does not exist\n";
    throw std::invalid_argument("Param " + param + " does not exist");
  }
  return std::stof(this->args_for_run[param]);
}

std::string Experiment::get_string_param(const std::string &param) {
//    std::cout << "Param count " << param << this->args_for_run.count(param) << " " << std::endl;
  if (this->args_for_run.count(param) == 0) {
    std::cout << "Param does not exist\n";
    throw std::invalid_argument("Param " + param + " does not exist");
  }
  return this->args_for_run[param];
}