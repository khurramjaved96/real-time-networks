import random
import numpy as np

from .base_agent import BaseAgent


class SarsaControlAgent(BaseAgent):
    """Implements the sarsa control agent"""

    def __init__(self):
        pass

    def train(self, env, model, timesteps, epsilon, gamma, lmbda, logger, args):
        """Train the agent
        Args:
            _
        Return:
            -
        """
        done = True
        running_eps_reward = -2000
        eps_count = 0
        eps_rewards = 0
        obs = env.reset()
        for t in range(timesteps):
            model.set_input_values(obs)
            model.step()
            qvalues = model.read_output_values()

            if done:  # new episode starts
                if random.random() <= epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qvalues)

            next_obs, reward, done, info = env.step(action)
            no_grad = np.ones(len(qvalues))

            if done:
                new_target = reward
            else:
                next_qvalues = model.forward_pass_without_side_effects(next_obs)
                if random.random() <= epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(next_qvalues)
                new_target = reward + gamma * next_qvalues[next_action]

            no_grad[action] = 0
            err = model.introduce_targets(new_target, gamma, lmbda, no_grad)

            obs = next_obs
            action = next_action
            eps_rewards += reward

            if done:
                obs = env.reset()
                running_eps_reward = 0.01 * eps_rewards + 0.99 * running_eps_reward
                # print(env.expanding_size)
                print(t, eps_count, eps_rewards, running_eps_reward, qvalues)
                eps_rewards = 0
                eps_count += 1
                model.reset_trace()
        env.close()
