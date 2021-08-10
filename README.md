[![Test Cases](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cmake.yml/badge.svg?branch=step_size_adaptation&event=push)](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cmake.yml) [![cpplint](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cpplint.yml/badge.svg?event=push)](https://github.com/khurramjaved96/real-time-networks/actions/workflows/cpplint.yml) [![codecov](https://codecov.io/gh/khurramjaved96/continually-adapting-networks/branch/development/graph/badge.svg?token=3YDYPKYSKO)](https://codecov.io/gh/khurramjaved96/real-time-networks)

# Real-time Neural Networks
Implementation of Real-time Neural Networks for agent-state construction with a focus on constructive approaches

## Requirements
In order to run this project, you'll need the following things installed:
* GCC (version 9.3.0 and above)
* make
* A C++ compiler (C++17 and above)
* [MariaDB](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/) (and a C++ connector for MariaDB
  found [here](https://mariadb.com/kb/en/mariadb-connector-c/))
  
For the tests, you'll need Python installed together with `pytorch` and `autograd`.

To compile this project locally, you'll have to link your MariaDB C++ connector locally, as demonstrated
in `CMakeLists.txt`. You'll also need to uncomment essentially everything under the 
comment "For running locally".

## Instructions for python extension
* Install pybind11
* From the project's root directory, use `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`
* Use CMakeListsPy.txt to compile after adjusting the `include_directories` inside.
* Train using `python train_CAN.py --run-id 0 --algo dqn --env PongNoFrameskip-v4 --no-render --deterministic --n-timesteps 20000000`
