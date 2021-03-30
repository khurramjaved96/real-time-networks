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
    std::map<std::string, std::string> args_for_run;
    std::string output_dir;
    Database d = Database();

    static std::vector<int> frequency_of_params(std::map <std::string, std::vector<std::string>> &args);
//    std::map<std::string, std::string> get_args_for_run(int run, std::map<std::string, std::vector<std::string>> args, std::vector<int>);

public:
    int run;
    std::string database_name;
//    Experiment(std::string name, std::map<std::string, std::string> args, std::string output_dir, bool sql, int rank);
    Experiment(int name, char *argv[]);
    static std::map<std::string, std::vector<std::string>> parse_params(int total_prams, char *pram_list[]);
//    bool make_table(std::string table_name, std::vector<std::string> column_names, std::vector<std::string> data_types,  std::vector<std::string> primary_key);
//    bool insert_values(std::string table_name, std::vector<std::string> column_names, std::vector<std::vector<std::string>> value_lists);
};


#endif //BENCHMARKS_EXPERIMENT_H
