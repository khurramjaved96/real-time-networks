[![Test Cases](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cmake.yml/badge.svg?branch=step_size_adaptation&event=push)](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cmake.yml) [![cpplint](https://github.com/khurramjaved96/continually-adapting-networks/actions/workflows/cpplint.yml/badge.svg?event=push)](https://github.com/khurramjaved96/real-time-networks/actions/workflows/cpplint.yml) [![codecov](https://codecov.io/gh/khurramjaved96/real-time-networks/branch/development/graph/badge.svg?token=3YDYPKYSKO)](https://codecov.io/gh/khurramjaved96/real-time-networks)

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

## Instructions for running experiments in Python
* Install packages `pip install -r requirements.txt`
* Install `pybind11` and adjust the pybind directory in `CMakeListsPy.txt` (recommended to install it as subdir in this project)
* Use `CMakeListsPy.txt` to compile

### Mountain Car control experiment
* Train using: `Python train_gym.py --env-max-step-per-episode 1000 -t control --tilecoding 1`

### Atari prediction experiments
* From the project's root directory, use `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo` to get the pretrained expert agents
* Train using: `python train_gym.py --net imprintingAtari --imprinting-mode random --env PongNoFrameskip-v4 --binning 1 --gamma 0.5 --meta-step-size 0.01`
