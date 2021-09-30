import random
import argparse
import glob
import importlib
import os
import sys
from datetime import datetime
from time import sleep

import gym
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,"rl-baselines3-zoo")

import FlexibleNN
from FlexibleNN import Metric, Database
from python_scripts.utils.utils import get_types
from python_scripts.utils.state_feature.state_feature_util import TileCoder
from python_scripts.utils.logging_manager import LoggingManager
from python_scripts.utils.tilecoding_wrapper import TileCodedObservation
from python_scripts.utils.image_binning_wrapper import BinnedObservation
from python_scripts.agents.sarsa_control_agent import SarsaControlAgent
from python_scripts.agents.sarsa_prediction_agent import SarsaPredictionAgent
from python_scripts.agents.sarsa_continuous_prediction_agent import SarsaContinuousPredictionAgent
from python_scripts.agents.mountaincar_fixed_agent import MountainCarFixed
from python_scripts.agents.baselines_expert_agent import BaselinesExpert


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

    # make sure run_ids dont overlap when using parallel
    sleep(random.random()*10)

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "-t", "--task", help="task type: prediction, control (default: prediction)", default="prediction", type=str,)
    parser.add_argument( "--net", help="network type: LFA, expandingLFA, imprinting (default: expandingLFA)", type=str, default="expandingLFA",)
    parser.add_argument( "--n-timesteps", help="number of timesteps", default=1000000, type=int)

    parser.add_argument( "--env", help="environment ID", type=str, default="MountainCar-v0")
    parser.add_argument( "--env-max-step-per-episode", help="Max number of timesteps per episode (default:0, mountaincar:2000)", default=0, type=int,)

    parser.add_argument( "--binning", help="Use binned features (0: dont use, 1: use)", default=0, type=int,)
    parser.add_argument( "--binning-n-bins", help="Number of binning bins to use (default: 10)", default=10, type=int,)

    parser.add_argument( "--tilecoding", help="Use tilecoded features (0: dont use, 1: use)", default=0, type=int,)
    parser.add_argument( "--tilecoding-n-tilings", help="Number of tilecoding tilings to use (default: 8)", default=8, type=int,)
    parser.add_argument( "--tilecoding-n-tiles", help="Number of tilecoding tiles per dim (default: 8)", default=8, type=int,)

    parser.add_argument( "--net-width", help="initial width of the network (only for net:imprintingWide)", default=0, type=int)
    parser.add_argument( "--net-prune-prob", help="pruning prob (per step) for the weights after they have matured", default=0.01, type=float)
    parser.add_argument( "--imprinting-max-bound-range", help="max range for the random bounds that are found around the random center", default=0.1, type=float)
    parser.add_argument( "--use-imprinting", help="Use imprinted features instead of random (0: dont use, 1: use)", default=1, type=int,)
    parser.add_argument( "--imprinting-err-thresh", help="If error_trace-current_error > thresh, do imprinting", default=0.1, type=float)

    parser.add_argument("--step-size", help="step size", default=0.01, type=float)
    parser.add_argument( "--meta-step-size", help="tidbd step size", default=1e-3, type=float)
    parser.add_argument("--gamma", help="gamma", default=0.99, type=float)
    parser.add_argument("--lmbda", help="lambda", default=0.99, type=float)
    parser.add_argument( "--epsilon", help="exploration epsilon", default=0.10, type=float)

    args = parser.parse_args()

    if args.db == "":
        print("db name not provided. Not logging results")
        episodic_metrics = None
        neuron_metrics = None
        synapse_metrics = None
        prediction_metrics = None
        bounded_unit_metrics = None
    else:
        args.db = "hshah1_" + args.db
        Database().create_database(args.db)
        run_metric = Metric(args.db, "runs", list(vars(args).keys()), get_types(list(vars(args).values())), ["run_id"])
        run_metric.add_value([str(v) for v in list(vars(args).values())])

        episodic_metrics = Metric(
            args.db,
            "episodic_metrics",
            ["run_id", "episode", "timestep", "MSRE", "running_MSRE", "error"], ["int", "int", "int", "real", "real", "real"],
            ["run_id", "episode"],
        )
        neuron_metrics = Metric(
            args.db,
            "neuron_metrics",
            ["run_id", "episode", "timestep", "neuron_id", "value", "avg_value", "neuron_utility"],
            ["int", "int", "int", "int", "real", "real", "real"],
            ["run_id", "timestep", "neuron_id"],
        )
        synapse_metrics = Metric(
            args.db,
            "synapse_metrics",
            ["run_id", "episode", "timestep", "synapse_id", "weight", "step_size", "synapse_utility"],
            ["int", "int", "int", "int", "real", "real", "real"],
            ["run_id", "timestep", "synapse_id"],
        )
        prediction_metrics = Metric(
            args.db,
            "prediction_metrics",
            ["run_id", "episode", "timestep", "MSRE", "prediction", "return_target", "return_error", "synapse_ids"],
            ["int", "int", "int", "real", "real", "real", "real", "JSON"],
            ["run_id", "episode", "timestep"],
        )
        bounded_unit_metrics= Metric(
            args.db,
            "bounded_unit_metrics",
            ["run_id", "episode", "timestep", "count_active"],
            ["int", "int", "int", "int"],
            ["run_id", "timestep"],
        )
    # fmt: on

    if args.net == "expandingLFA":
        assert args.tilecoding, f"expandingLFA can only be used with tilecoding"

    if args.env == "PongNoFrameskip-v4":
        expert_agent = BaselinesExpert(seed=args.seed, env_id=args.env)
        env = expert_agent.env
        input_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        print(f"Using input size of {input_size}")
    else:
        env = gym.make(args.env)
        input_size = env.observation_space.shape[0]

    if args.task == "control":
        output_size = env.action_space.n
    else:
        output_size = 1

    if args.env == "CartPole-v1":  # cartpole input range is too large to use directly
        input_range = [(-3, 3), (-3.5, 3.5), (-0.25, 0.25), (-3.5, 3.5)]
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
    if args.binning:
        assert not args.tilecoding, f"not to be used with tc"
        env = BinnedObservation(env, args.binning_n_bins)
        input_size = input_size * args.binning_n_bins

    if args.env == "PongNoFrameskip-v4":
        assert args.binning, f"Should use binning with Pong"

    if args.env_max_step_per_episode:  # just wrapper thing
        if args.tilecoding:
            env.env._max_episode_steps = args.env_max_step_per_episode
        else:
            env._max_episode_steps = args.env_max_step_per_episode

    set_random_seed(args.seed, env)

    if args.net == "LFA":
        model = FlexibleNN.LinearFunctionApproximator(
            input_size,
            output_size,
            args.step_size,
            args.meta_step_size,
            True,
        )
    elif args.net == "expandingLFA":
        model = FlexibleNN.ExpandingLinearFunctionApproximator(
            input_size,
            output_size,
            args.tilecoding_n_tilings * 2,
            args.step_size,
            args.meta_step_size,
            True,
        )
    elif args.net == "imprintingWide":
        model = FlexibleNN.ImprintingWideNetwork(
            input_size,
            output_size,
            args.net_width,
            input_range,
            1-args.net_prune_prob,
            args.imprinting_max_bound_range,
            args.step_size,
            args.meta_step_size,
            True,
            args.seed,
            bool(args.use_imprinting),
        )
    elif args.net == "imprintingAtari":
        assert args.env in ["PongNoFrameskip-v4"]
        assert args.net_width == 0, f"net width not implemented"
        model = FlexibleNN.ImprintingAtariNetwork(
            input_size,
            output_size,
            args.net_width,
            args.step_size,
            args.meta_step_size,
            True,
            args.seed,
            bool(args.use_imprinting),
            env.observation_space.shape[0],
            env.observation_space.shape[1],
            args.binning_n_bins,
        )
    else:
        raise NotImplementedError

    logger = LoggingManager(
        log_to_db=(args.db != ""),
        run_id=args.run_id,
        model=model,
        commit_frequency=10000,
        episodic_metrics=episodic_metrics,
        neuron_metrics=neuron_metrics,
        synapse_metrics=synapse_metrics,
        prediction_metrics=prediction_metrics,
        bounded_unit_metrics=bounded_unit_metrics,
    )

    if args.task == "control":
        agent = SarsaControlAgent()
    elif args.task == "prediction":
        if args.env == "MountainCar-v0":
            expert_agent = MountainCarFixed()
            agent = SarsaPredictionAgent(expert_agent)
        elif args.env == "PongNoFrameskip-v4":
            agent = SarsaContinuousPredictionAgent(expert_agent)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    agent.train(env, model, args.n_timesteps, args.epsilon, args.gamma, args.lmbda, logger, args)


#    if args.db:
#        bound_replacement_metrics = Metric(
#            args.db,
#            "bound_replacement_metrics",
#            ["run_id", "neuron_id", "neuron_age", "neuron_utility", "output_weight", "num_times_replaced" ], ["int", "int", "int", "real", "real", "int"],
#            ["run_id", "neuron_id"],
#        )
#        logger.log_synapse_replacement(bound_replacement_metrics)
#
    logger.commit_logs()

if __name__ == "__main__":
    main()
