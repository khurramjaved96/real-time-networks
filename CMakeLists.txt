cmake_minimum_required(VERSION 3.14)
project(binding LANGUAGES CXX CUDA)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread" )

add_library(binding SHARED test.cpp)

