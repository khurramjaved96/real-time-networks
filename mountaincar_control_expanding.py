import random
import argparse
import glob
import importlib
import os
import sys

import gym
import numpy as np
import matplotlib.pyplot as plt

import FlexibleNN
from state_feature.state_feature_util import TileCoder

def set_random_seed(seed: int, env: gym.wrappers.time_limit.TimeLimit) -> None:
    """
    Seed the different random generators.

    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)


def get_tc_onehot(obs, tc, tc_tracker = []):
    indices = tc.transform(obs)
    if tc_tracker != []:
        tc_tracker[indices] += 1
    oh_vec = np.zeros(tc.total_tiles)
    oh_vec[indices] = 1
    return oh_vec


def get_expanding_tc_onehot(obs, tc, max_inp_size, tc_tracker = []):
    indices = tc.transform(obs)
    curr_inp_size = max(indices) + 1
    if curr_inp_size > max_inp_size:
        max_inp_size = curr_inp_size
    if tc_tracker != []:
        tc_tracker[indices] += 1
    oh_vec = np.zeros(max_inp_size)
    oh_vec[indices] = 1
    return oh_vec, max_inp_size


def main():  # noqa: C901
    env = gym.make("MountainCar-v0")
    num_tilings = 8

    #cartpole
    #tc_range = [(-3,3), (-3.5, 3.5), (-0.25, 0.25), (-3.5, 3.5)]
    #tc = TileCoder(env.observation_space.shape[0], num_tilings, tc_range, [4]*env.observation_space.shape[0])


    #tc_range = np.array(tuple(zip(env.observation_space.low, env.observation_space.high))) #correct way

    #mountaincar
    tc = TileCoder(2, 8, [(-1.2, 0.6), (-0.07, 0.07)], [8, 8])


    #input_size = env.observation_space.shape[0]
    input_size = tc.total_tiles
    output_size = env.action_space.n
    seed = 1
    step_size = 0.01
    #step_size = 0.0000000001
    step_size = step_size / num_tilings
    meta_step_size = 1e-3
    gamma = 0.99
    lmbda = 0.99
    epsilon = 0.1
    timesteps = 1000000
    env._max_episode_steps = 2000
    max_inp_size = num_tilings*2
    done = True

    set_random_seed(seed, env)

    #model = FlexibleNN.LinearFunctionApproximator(input_size, output_size, step_size, meta_step_size, True)
    model = FlexibleNN.ExpandingLinearFunctionApproximator(input_size, output_size, max_inp_size, step_size, meta_step_size, True)

    tc_tracker = np.zeros(tc.total_tiles)
    running_eps_reward = -2000

    eps_count = 0
    eps_rewards = 0
    obs = env.reset()
    for t in range(timesteps):
        oh_obs, max_inp_size = get_expanding_tc_onehot(obs, tc, max_inp_size, tc_tracker)
        model.set_input_values(oh_obs)
        model.step()
        qvalues = model.read_output_values()
        #print(qvalues)

        if (done): # new episode starts
            if (random.random() <= epsilon):
                action = env.action_space.sample()
            else:
                action = np.argmax(qvalues)

        next_obs, reward, done, info = env.step(action)
        no_grad = np.ones(output_size)

        if done:
            new_target = reward
        else:
            oh_obs, max_inp_size = get_expanding_tc_onehot(next_obs, tc, max_inp_size)
            next_qvalues = model.forward_pass_without_side_effects(oh_obs)
            if (random.random() <= epsilon):
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(next_qvalues)
            new_target = reward + gamma * next_qvalues[next_action]
        #print(new_target)

        no_grad[action] = 0
        err = model.introduce_targets(new_target, gamma, lmbda, no_grad)
        #print(err)

        obs = next_obs
        action = next_action
        eps_rewards += reward

        if done:
            obs = env.reset()
            running_eps_reward = 0.01 * eps_rewards + 0.99 * running_eps_reward
            print(t, eps_count, eps_rewards, running_eps_reward, qvalues, max_inp_size)
            eps_rewards = 0
            eps_count += 1
            model.reset_trace()

    env.close()
    from IPython import embed; embed()



if __name__ == "__main__":
    main()
