//
// Created by Khurram Javed on 2021-03-16.
//

#include <iostream>
#include <vector>
#include <iomanip>
#include "../include/utils.h"


void print_vector(std::vector<float> const &v) {
    std::cout << "[";
    int counter = 0;
    for (auto i = v.begin(); i != v.end(); ++i) {
        std::cout << " " << std::setw(3)  <<  *i << ",";
        if (counter % 10 == 9) std::cout << "\n";
        counter++;
    }
    std::cout << "]\n";
}

void print_vector(std::vector<int> const &v) {
    std::cout << "[";
    int counter = 0;
    for (auto i = v.begin(); i != v.end(); ++i) {
        std::cout << " "  << std::setw(3) << *i << ",";
        if (counter % 100 == 99) std::cout << "\n";
        counter++;
    }
    std::cout << "]\n";
}


void print_matrix(std::vector<std::vector<int>> const &v) {
    int counter = 0;
    for(int temp=0; temp<v.size(); temp++) {
        if (temp > 50)
        {   std::cout << "Truncating output\n";
            break;
        }
        std::cout << "[";
        int counter_inner = 0;
        for (auto i = v[temp].begin(); i != v[temp].end(); ++i) {
            if(counter_inner > 50){
                std::cout << ", ... , ]";
                break;
            }
            std::cout << " " << std::setw(3)  << *i << ",";
//            if (counter % 50 == 49) std::cout << "\n";
            counter++;
            counter_inner++;
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}