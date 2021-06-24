# Continually Adapting Networks
Implementation of dynamic NNs for agent-state construction using genetate and test with gradients. 

## Requirements
In order to run this project, you'll need the following things installed:
* GCC (version 10 and above)
* make
* A C++ compiler (C++17 and above)
* [MariaDB](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/) (and a C++ connector for MariaDB
  found [here](https://mariadb.com/kb/en/mariadb-connector-c/))
  
For the tests, you'll need Python installed together with `pytorch`.

To compile this project locally, you'll have to link your MariaDB C++ connector locally, as demonstrated
in `CMakeLists.txt`. You'll also need to uncomment essentially everything under the 
comment "For running locally".