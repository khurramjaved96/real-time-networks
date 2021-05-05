//
// Created by Khurram Javed on 2021-03-14.
//

#ifndef BENCHMARKS_EXPERIMENT_H
#define BENCHMARKS_EXPERIMENT_H

#include <string>
#include <vector>
#include <map>
#include "Database.h"

class Experiment {
//    std::string name;
    std::map<std::string, std::vector<std::string>> args;

    std::string output_dir;
    Database d = Database();

    static std::vector<int> frequency_of_params(std::map <std::string, std::vector<std::string>> &args);
//    std::map<std::string, std::string> get_args_for_run(int run, std::map<std::string, std::vector<std::string>> args, std::vector<int>);

public:
    std::map<std::string, std::string> args_for_run;
    int run;
    std::string database_name;
    Experiment(int name, char *argv[]);
    static std::map<std::string, std::vector<std::string>> parse_params(int total_prams, char *pram_list[]);
    int get_int_param(const std::string&);
    float get_float_param(const std::string&);
    bool get_bool_param(const std::string& param);
};


#endif //BENCHMARKS_EXPERIMENT_H
