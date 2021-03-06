import random
import argparse
import glob
import importlib
import os
import sys
from datetime import datetime, timedelta
from time import sleep
from timeit import default_timer as timer

import gym
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "rl-baselines3-zoo")

import FlexibleNN
from FlexibleNN import Metric, Database
from src_python.utils.utils import get_types
from src_python.utils.state_feature.state_feature_util import TileCoder
from src_python.utils.logging_manager import LoggingManager
from src_python.utils.tilecoding_wrapper import TileCodedObservation
from src_python.utils.image_binning_wrapper import BinnedObservation
from src_python.models.linear_model import LinearModel
from src_python.envs.classical_conditioning_benchmarks import (
    TraceConditioning,
    TracePatterning,
)
from src_python.agents.baselines_expert_agent import BaselinesExpert
from src_python.agents.mountaincar_fixed_agent import MountainCarFixed
from src_python.agents.sarsa_control_agent import SarsaControlAgent
from src_python.agents.sarsa_prediction_agent import SarsaPredictionAgent
from src_python.agents.sarsa_continuous_prediction_agent import (
    SarsaContinuousPredictionAgent,
)
from src_python.agents.torch_sarsa_continuous_prediction_agent import (
    TorchSarsaContinuousPredictionAgent,
)


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
    sleep(random.random() * 10)

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
    parser.add_argument( "--expert-policy-deterministic", help="Whether the expert policy is deterministic (1: greedy, 0: eps_greedy with eps=args.epsilon", default=1, type=int,)

    parser.add_argument( "--binning", help="Use binned features (0: dont use, 1: use)", default=0, type=int,)
    parser.add_argument( "--binning-n-bins", help="Number of binning bins to use (default: 10)", default=10, type=int,)

    parser.add_argument( "--tilecoding", help="Use tilecoded features (0: dont use, 1: use)", default=0, type=int,)
    parser.add_argument( "--tilecoding-n-tilings", help="Number of tilecoding tilings to use (default: 8)", default=8, type=int,)
    parser.add_argument( "--tilecoding-n-tiles", help="Number of tilecoding tiles per dim (default: 8)", default=8, type=int,)

    parser.add_argument( "--net-width", help="initial width of the network (only for net:imprintingWide)", default=0, type=int)
    parser.add_argument( "--net-prune-prob", help="pruning prob (per step) for the weights after they have matured", default=0.01, type=float)
    parser.add_argument( "--imprinting-max-bound-range", help="max range for the random bounds that are found around the random center", default=0.1, type=float)
    parser.add_argument( "--use-optical-flow-state", help="Add optical flow information as a state", default=0, type=int,)
    parser.add_argument( "--use-imprinting", help="Use imprinted features instead of random (0: dont use, 1: use)", default=1, type=int,)
    parser.add_argument( "--imprinting-err-thresh", help="If error_trace-current_error > thresh, do imprinting", default=0.1, type=float)
    parser.add_argument( "--imprinting-mode", help="Imprinting mode to use (random, optical_flow: default)", default="optical_flow", type=str,)
    parser.add_argument( "--imprinting-max-prob", help="Max percentage of interesting features to imprint on. Used as U[0,imprinting-max-prob]", default=1, type=float)
    parser.add_argument( "--imprinting-random-prob", help="Prob at each step to generate a feature regardless of the error (default:0)", default=0, type=float)
    parser.add_argument( "--imprinting-only-single-layer", help="Restrict the feature generation to let network stay single layered forever", default=0, type=int,)

    parser.add_argument( "--utility-to-keep", help="Utility to keep (used to set the number of max synapses)", default=0.0001, type=float,)
    parser.add_argument( "--linear-drinking-age", help="maturity age: drinking_age*4. This is only for input features.", default=5000, type=int,)
    parser.add_argument( "--linear-synapse-local-utility-trace-decay", help="Rate at which it decays (see code), only for linear/input synapses.", default=0.9999, type=float,)

    parser.add_argument("--step-size", help="step size", default=0.01, type=float)
    parser.add_argument( "--meta-step-size", help="tidbd step size", default=1e-3, type=float)
    parser.add_argument("--gamma", help="gamma", default=0.99, type=float)
    parser.add_argument("--lmbda", help="lambda", default=0.99, type=float)
    parser.add_argument( "--epsilon", help="exploration epsilon", default=0.10, type=float)

    # params for animal state experiments
    parser.add_argument("--num-CS", type=int, default=1)
    parser.add_argument("--num-US", type=int, default=1)
    parser.add_argument("--num-dist", type=int, default=10)
    parser.add_argument("--num-activation-patterns", type=int, default=10)
    parser.add_argument("--prob-activation-patterns", type=float, default=0.3)
    parser.add_argument("--ISI-interval", type=str, default="7,13")
    parser.add_argument("--ITI-interval", type=str, default="80,120")
    parser.add_argument("--len-CS", type=int, default=4)
    parser.add_argument("--len-US", type=int, default=2)
    parser.add_argument("--len-dist", type=int, default=4)
    parser.add_argument("--CS-noise", type=float, default=0)


    args = parser.parse_args()

    episodic_metrics = None
    neuron_metrics = None
    synapse_metrics = None
    prediction_metrics = None
    bounded_unit_metrics = None
    imprinting_metrics = None
    linear_feature_metrics = None
    if args.db == "":
        print("db name not provided. Not logging results")
    else:
        args.db = "hshah1_" + args.db
        Database().create_database(args.db)
        run_metric = Metric(args.db, "runs", list(vars(args).keys()), get_types(list(vars(args).values())), ["run_id"])
        run_metric.add_value([str(v) for v in list(vars(args).values())])

        run_state_metric = Metric(
            args.db,
            "run_states",
            ["run_id", "comment", "state", "timestep", "episode", "MSRE", "running_MSRE", "n_features", "n_synapses", "run_time"],
            ["int", "VARCHAR(80)", "VARCHAR(40)", "int", "int", "real", "real", "int", "int", "VARCHAR(60)"],
            ["run_id"],
        )
        episodic_metrics = Metric(
            args.db,
            "episodic_metrics",
            ["run_id", "episode", "timestep", "MSRE", "running_MSRE", "error"],
            ["int", "int", "int", "real", "real", "real"],
            ["run_id", "episode", "timestep"],
        )
        prediction_metrics = Metric(
            args.db,
            "prediction_metrics",
            ["run_id", "episode", "timestep", "MSRE", "prediction", "return_target", "return_error", "synapse_ids"],
            ["int", "int", "int", "real", "real", "real", "real", "JSON"],
            ["run_id", "episode", "timestep"],
        )
        if not args.net in ['torchLinear']:
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
            bounded_unit_metrics= Metric(
                args.db,
                "bounded_unit_metrics",
                ["run_id", "episode", "timestep", "count_active"],
                ["int", "int", "int", "int"],
                ["run_id", "timestep"],
            )
            imprinting_metrics = Metric(
                args.db,
                "imprinting_metric",
                ["run_id", "episode", "timestep", "neuron_id", "imprinted_on_id", "outgoing_weight", "step_size", "age", "neuron_utility"],
                ["int", "int", "int", "int", "int", "real", "real", "int", "real"],
                ["run_id", "timestep", "neuron_id", "imprinted_on_id"],
            )
            linear_feature_metrics = Metric(
                args.db,
                "linear_feature_metric",
                ["run_id", "episode", "timestep", "neuron_id", "outgoing_weight", "step_size", "neuron_utility", "synapse_utility", "synapse_utility_to_distribute"],
                ["int", "int", "int", "int", "real", "real", "real", "real", "real"],
                ["run_id", "timestep", "neuron_id"],
            )
    # fmt: on

    atari_envs = ["PongNoFrameskip-v4"]
    animal_state_envs = ["TraceConditioning", "NoisyPatterning", "TracePatterning"]
    if args.env in animal_state_envs:
        args.ISI_interval = [int(x) for x in args.ISI_interval.split(",")]
        args.ITI_interval = [int(x) for x in args.ITI_interval.split(",")]
        args.gamma = 1 - 1 / np.mean(args.ISI_interval)
        print(f"Provided gamma not being used. Using gamma: {args.gamma}")

    if args.net == "expandingLFA":
        assert args.tilecoding, f"expandingLFA can only be used with tilecoding"

    if args.use_optical_flow_state:
        assert args.binning, f"optical flow state only implmnted with binning"
        assert args.env in atari_envs, f"optical flow state only tested with atari"

    if args.env == "TraceConditioning":
        env = TraceConditioning(
            seed=args.seed,
            ISI_interval=args.ISI_interval,
            ITI_interval=args.ITI_interval,
            gamma=args.gamma,
            num_distractors=args.num_dist,
            activation_lengths={
                "CS": args.len_CS,
                "US": args.len_US,
                "distractor": args.len_dist,
            },
        )
        input_size = args.num_CS + args.num_US + args.num_dist
    elif args.env == "TracePatterning":
        env = TracePatterning(
            seed=args.seed,
            ISI_interval=args.ISI_interval,
            ITI_interval=args.ITI_interval,
            gamma=args.gamma,
            num_CS=args.num_CS,
            num_activation_patterns=args.num_activation_patterns,
            prob_activation_patterns=args.prob_activation_patterns,
            num_distractors=args.num_dist,
            activation_lengths={
                "CS": args.len_CS,
                "US": args.len_US,
                "distractor": args.len_dist,
            },
            noise=args.CS_noise,
        )
        input_size = args.num_CS + args.num_US + args.num_dist

    elif args.env == "PongNoFrameskip-v4":
        expert_agent = BaselinesExpert(
            seed=args.seed,
            env_id=args.env,
            deterministic=bool(args.expert_policy_deterministic),
            exploration_rate=args.epsilon,
        )
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
        assert not args.env in animal_state_envs, f"tc not to be used with animal envs"
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
        assert not args.tilecoding, f"binning not to be used with tc"
        assert (
            not args.env in animal_state_envs
        ), f"binning not to be used with animal envs"
        env = BinnedObservation(env, args.binning_n_bins, args.use_optical_flow_state)
        input_size = input_size * args.binning_n_bins
        if args.use_optical_flow_state:
            input_size += env.observation_space.shape[0] * env.observation_space.shape[1]

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
            1 - args.net_prune_prob,
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
            args.imprinting_max_prob,
            bool(args.imprinting_only_single_layer),
            bool(args.use_optical_flow_state),
            args.linear_drinking_age,
            args.linear_synapse_local_utility_trace_decay,
            args.utility_to_keep,
        )
    elif args.net == "torchLinear":
        model = LinearModel(input_size, output_size, args.step_size, False)
    else:
        raise NotImplementedError

    logger = LoggingManager(
        log_to_db=(args.db != ""),
        run_id=args.run_id,
        model=model,
        commit_frequency=5000,
        episodic_metrics=episodic_metrics,
        neuron_metrics=neuron_metrics,
        synapse_metrics=synapse_metrics,
        prediction_metrics=prediction_metrics,
        imprinting_metrics=imprinting_metrics,
        linear_feature_metrics=linear_feature_metrics,
    )

    if args.task == "control":
        agent = SarsaControlAgent()
    elif args.task == "prediction":
        if args.env == "MountainCar-v0":
            expert_agent = MountainCarFixed()
            agent = SarsaPredictionAgent(expert_agent)
        elif args.env == "PongNoFrameskip-v4" and args.net not in ["torchLinear"]:
            agent = SarsaContinuousPredictionAgent(expert_agent)
        elif args.env == "PongNoFrameskip-v4" and args.net in ["torchLinear"]:
            agent = TorchSarsaContinuousPredictionAgent(expert_agent)
        elif args.env in animal_state_envs:
            agent = SarsaContinuousPredictionAgent(None)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    start = timer()
    state_comment = "finished"
    try:
        agent.train(
            env,
            model,
            args.n_timesteps,
            args.epsilon,
            args.gamma,
            args.lmbda,
            logger,
            args,
        )
    except:
        state_comment = "killed"
        print("failed... quiting")
    finally:
        if args.db != "":
            run_state_metric.add_value(
                [
                    str(v)
                    for v in [
                        args.run_id,
                        args.comment,
                        state_comment,
                        agent.timestep,
                        agent.episode,
                        agent.MSRE,
                        agent.running_MSRE,
                        len(model.imprinted_features),
                        len(model.all_synapses),
                        str(timedelta(seconds=timer() - start)),
                    ]
                ]
            )
        logger.commit_logs()


if __name__ == "__main__":
    main()
