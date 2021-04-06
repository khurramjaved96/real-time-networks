//
// Created by Khurram Javed on 2021-03-14.
//

#ifndef BENCHMARKS_EXPERIMENT_H
#define BENCHMARKS_EXPERIMENT_H

#ifndef DATE_H
#define DATE_H
#include <string>
#include <vector>
#include <map>

class Date
{
private:
    int m_year;
    int m_month;
    int m_day;

public:
    Date(int year, int month, int day);

    void SetDate(int year, int month, int day);

    int getYear() { return m_year; }
    int getMonth() { return m_month; }
    int getDay()  { return m_day; }
};

#endif


class Experiment {
    std::string name;
    std::map<std::string, std::string> args;
    std::string output_dir = output_dir;

public:
    Experiment(std::string name, std::map<std::string, std::string> args, std::string output_dir, bool sql, int rank);
    bool make_table(std::string table_name, std::vector<std::string> column_names, std::vector<std::string> data_types,  std::vector<std::string> primary_key);
    bool insert_values(std::string table_name, std::vector<std::string> column_names, std::vector<std::vector<std::string>> value_lists);
};


#endif //BENCHMARKS_EXPERIMENT_H
