import random
import argparse
import glob
import importlib
import os
import sys
from datetime import datetime

import gym
import numpy as np
import matplotlib.pyplot as plt

import FlexibleNN
from FlexibleNN import Metric, Database
from python_scripts.utils.utils import get_types
from python_scripts.utils.state_feature.state_feature_util import TileCoder
from python_scripts.utils.tilecoding_wrapper import TileCodedObservation
from python_scripts.agents.sarsa_control_agent import SarsaControlAgent
from python_scripts.agents.sarsa_prediction_agent import SarsaPredictionAgent
from python_scripts.agents.mountaincar_fixed_agent import MountainCarFixed


def set_random_seed(seed: int, env: gym.wrappers.time_limit.TimeLimit) -> None:
    """
    Seed the different random generators.
    :param seed:
    :param env: gym env
    """
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)


def main():  # noqa: C901

    # These params work for mountaincar (not sweeped yet), just set env-max-step-per-episode=2000
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%m%d%H%M%S"), type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "-t", "--task", help="task type: prediction, control (default: prediction)", default="prediction", type=str,)
    parser.add_argument( "--net", help="network type: LFA, expandingLFA, imprinting (default: expandingLFA)", type=str, default="expandingLFA",)
    parser.add_argument( "--n-timesteps", help="number of timesteps", default=1000000, type=int)

    parser.add_argument( "--env", help="environment ID", type=str, default="MountainCar-v0")
    parser.add_argument( "--env-max-step-per-episode", help="Max number of timesteps per episode (default:0, mountaincar:2000)", default=0, type=int,)

    parser.add_argument( "--tilecoding", help="Use tilecoded features (0: dont use, 1: use)", default=1, type=int,)
    parser.add_argument( "--tilecoding-n-tilings", help="Number of tilecoding tilings to use (default: 8)", default=8, type=int,)
    parser.add_argument( "--tilecoding-n-tiles", help="Number of tilecoding tiles per dim (default: 8)", default=8, type=int,)

    parser.add_argument("--step-size", help="step size", default=0.01, type=float)
    parser.add_argument( "--meta-step-size", help="tidbd step size", default=1e-3, type=float)
    parser.add_argument("--gamma", help="gamma", default=0.99, type=float)
    parser.add_argument("--lmbda", help="lambda", default=0.99, type=float)
    parser.add_argument( "--epsilon", help="exploration epsilon", default=0.10, type=float)

    args = parser.parse_args()

    if args.db == "":
        print("db name not provided. Not logging results")
    else:
        args.db = "hshah1_" + args.db
        Database().create_database(args.db)
        run_metric = Metric(args.db, "runs", list(vars(args).keys()), get_types(list(vars(args).values())), ["run_id"])
        run_metric.add_value([str(v) for v in list(vars(args).values())])

        episodic_metrics = Metric(args.db, "episodic_metrics", ["run_id", "episode", "timestep", "MSRE", "error"],
                                  ["int", "int", "int", "real", "real"] ,["run_id", "episode"])
        neuron_metrics = Metric(args.db, "neuron_metrics", ["run_id", "episode", "timestep", "neuron_id", "value", "avg_value", "neuron_utility"],
                                ["int", "int", "int", "int", "real", "real", "real"], ["run_id", "timestep", "neuron_id"])
        synapse_metrics = Metric(args.db, "synapse_metrics", ["run_id", "episode", "timestep", "synapse_id", "weight", "step_size", "synapse_utility"],
                                 ["int", "int", "int", "int", "real", "real", "real"], ["run_id", "timestep", "synapse_id"])

    if args.net == "expandingLFA":
        assert args.tilecoding, f"expandingLFA can only be used with tilecoding"

    env = gym.make(args.env)
    input_size = env.observation_space.shape[0]
    if args.task == "control":
        output_size = env.action_space.n
    else:
        output_size = 1

    if args.env == "CartPole-v1":
        input_range = [(-3,3), (-3.5, 3.5), (-0.25, 0.25), (-3.5, 3.5)]
    else:
        input_range = np.array(tuple(zip(env.observation_space.low, env.observation_space.high)))

    if args.tilecoding:
        env = TileCodedObservation(
            env,
            env.observation_space.shape[0],
            args.tilecoding_n_tilings,
            input_range,
            [args.tilecoding_n_tiles] * env.observation_space.shape[0],
            is_expanding=(args.net == "expandingLFA"),
            expanding_initial_size=args.tilecoding_n_tilings * 2,
        )
        input_size = env.tc.total_tiles
        args.step_size /= args.tilecoding_n_tilings

    if args.env_max_step_per_episode:
        if args.tilecoding:
            env.env._max_episode_steps = args.env_max_step_per_episode
        else:
            env._max_episode_steps = args.env_max_step_per_episode

    set_random_seed(args.seed, env)

    if args.net == "LFA":
        model = FlexibleNN.LinearFunctionApproximator(input_size,
                                                      output_size,
                                                      args.step_size,
                                                      args.meta_step_size,
                                                      True)
    elif args.net == "expandingLFA":
        model = FlexibleNN.ExpandingLinearFunctionApproximator(input_size,
                                                               output_size,
                                                               args.tilecoding_n_tilings * 2,
                                                               args.step_size,
                                                               args.meta_step_size,
                                                               True)
    elif args.net == "imprintingWide":
        #TODO provide seed
        #TODO not used with tc
        model = FlexibleNN.ImprintingWideNetwork(input_size,
                                                 output_size,
                                                 5000,
                                                 input_range,
                                                 0.00001,
                                                 args.step_size,
                                                 args.meta_step_size,
                                                 True)
    else:
        raise NotImplementedError

    if args.task == "control":
        agent = SarsaControlAgent()
    else:
        if args.env == "MountainCar-v0":
            expert_agent = MountainCarFixed()
        else:
            raise NotImplementedError
        agent = SarsaPredictionAgent(expert_agent)

    agent.train(env, model, args.n_timesteps, args.epsilon, args.gamma, args.lmbda)
    from IPython import embed; embed()


if __name__ == "__main__":
    main()
