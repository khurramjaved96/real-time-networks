import argparse
import glob
import importlib
import os
import sys
sys.path.append("rl-baselines3-zoo")

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
import matplotlib.pyplot as plt

import FlexibleNN

def transform_obs(obs):
    return obs[0,:,:,-1].flatten()/255.0


def compute_return_error(cumulants, predictions, gamma):
    num_time_steps = len(cumulants)
    returns = np.zeros(num_time_steps)
    returns[-1] = cumulants[-1]
    for t in range(num_time_steps - 2, -1, -1):
        returns[t] = gamma * returns[t + 1] + cumulants[t]
    return_error = (predictions - returns) ** 2
    MSRE = return_error.mean()
    return MSRE, return_error, returns


def train_CAN(model, timestep, obs_vec, rewards_vec, step_size, gamma, lmbda, is_first_episode):
    predictions_vec = []
    for t in range(len(obs_vec) + 1):
        # for last step of the episode
        if t != len(obs_vec):
            obs = obs_vec[t]
        else:
            obs = np.zeros_like(obs_vec[-1])

        model.set_input_values(obs)
        model.step()

        if t > 0:
            prediction = model.read_output_values()[0]
            predictions_vec.append(prediction)
            if t != len(obs_vec):
                target = rewards_vec[t-1] + gamma * prediction
            else:
                target = rewards_vec[t-1]
            model.introduce_targets([target], gamma, lmbda, [0])

            if (t+timestep)%40000 == 0:
                print("x) Adding feature")
                for _ in range(5):
                    model.add_feature(step_size)

        if t == 0 and not is_first_episode:
            model.introduce_targets([0], 0, 0, [1])

    # Clear state
    for _ in range(20):
        model.set_input_values(np.zeros_like(obs_vec[-1]))
        model.step()
        model.introduce_targets([0], 0, 0, [1])
    model.reset_trace()

    return predictions_vec


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="run id for figures", default=0, type=int)
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-baselines3-zoo/rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    args = parser.parse_args()

    fig_path = 'py_figures/' + 'figs' + str(args.run_id)
    if os.path.exists(fig_path):
        print("run_id is already used. Choose a new one")
        exit()
    else:
        os.makedirs(fig_path)

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    input_size = 84*84
    gamma = 0.99
    lmbda = 0.98
    step_size = 0.000001
    model_CAN = FlexibleNN.ContinuallyAdaptingNetwork(step_size, args.seed, input_size, 1)

    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0

    is_first_episode = True
    running_MSRE = -1
    running_MSRE_vec = []
    obs_vec, rewards_vec = [], []


    try:
        for timestep in range(args.n_timesteps):
            obs_vec.append(transform_obs(obs))
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            rewards_vec.append(reward[0])
            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
                    state = None

                if done:
                    predictions_vec = train_CAN(model_CAN, timestep, obs_vec, rewards_vec, step_size, gamma, lmbda, is_first_episode)
                    MSRE, _, _ = compute_return_error(rewards_vec, predictions_vec, gamma)
                    if running_MSRE == -1:
                        running_MSRE = MSRE
                    else:
                        running_MSRE = 0.999 * running_MSRE + 0.001 * MSRE
                    running_MSRE_vec.append([timestep, running_MSRE])
                    print(f"Timestep: {timestep}, MSRE: {running_MSRE}, synapses: {model_CAN.get_total_synapses()}, neurons: {model_CAN.get_total_neurons()}")
                    obs_vec, rewards_vec = [], []
                    is_first_episode = False

            if timestep % 50000 == 0 and timestep > 1:
                plt_errs = np.asarray(running_MSRE_vec)
                plt.plot(plt_errs[:,0], plt_errs[:,1])
                plt.xlabel("timestep")
                plt.ylabel("running MSRE")
                plt.savefig('py_figures/' +  'figs' + str(args.run_id) + '/step_' + str(timestep) + '_plt.png')
                plt.clf()

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()



if __name__ == "__main__":
    main()
