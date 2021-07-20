[![CMake](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cmake.yml/badge.svg?branch=step_size_adaptation&event=push)](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cmake.yml)

# Continually Adapting Networks
Implementation of dynamic NNs for agent-state construction using genetate and test with gradients. 

## Requirements
In order to run this project, you'll need the following things installed:
* GCC (version 10 and above)
* make
* A C++ compiler (C++17 and above)
* [MariaDB](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/) (and a C++ connector for MariaDB
  found [here](https://mariadb.com/kb/en/mariadb-connector-c/))
  
For the tests, you'll need Python installed together with `pytorch` and `autograd`.

To compile this project locally, you'll have to link your MariaDB C++ connector locally, as demonstrated
in `CMakeLists.txt`. You'll also need to uncomment essentially everything under the 
comment "For running locally".

## Test cases
Test cases can be added in the tests directory. Create a new .h and .cpp file for the testcase and call the test case function in tests/test.cpp. The repo uses Travis-CI for Continuous Integration. 
