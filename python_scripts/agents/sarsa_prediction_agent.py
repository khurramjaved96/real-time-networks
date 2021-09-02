import sys
import random
import numpy as np

from .base_agent import BaseAgent
#sys.path.append("../utils")
#from utils import compute_return_error
from ..utils.utils import compute_return_error


class SarsaPredictionAgent(BaseAgent):
    """Implements the sarsa control agent

    """

    def __init__(self, expert_agent):
        self.expert_agent = expert_agent

    def train(self, env, model, timesteps, epsilon, gamma, lmbda):
        """ Train the agent
            Args:
                _
            Return:
                -
        """
        done = True
        #TODO running error
        eps_count = 0
        eps_rewards = []
        eps_predictions = []
        running_MSRE = -1
        obs = env.reset()
        for t in range(timesteps):

            model.set_input_values(obs)
            model.step()
            prediction = model.read_output_values()

            if done:  # new episode starts
                action = self.expert_agent.predict(obs)

            next_obs, reward, done, info = env.step(action)

            if done:
                new_target = reward
            else:
                bootstrap_prediction = model.forward_pass_without_side_effects(next_obs)
                bootstrap_action = self.expert_agent.predict(next_obs)
                new_target = reward + gamma * bootstrap_prediction[0]

            err = model.introduce_targets([new_target], gamma, lmbda)

            obs = next_obs
            action = bootstrap_action
            eps_rewards.append(reward)
            eps_predictions.append(prediction)

            if done:
                obs = env.reset()
                # first prediction is always 0 for now
                MSRE, return_error, return_target = compute_return_error(eps_rewards[1:], eps_predictions[1:], gamma)
                if eps_count == 0:
                    running_MSRE = MSRE
                else:
                    running_MSRE = 0.999 * running_MSRE + 0.001 * MSRE
                print(t, eps_count, MSRE, running_MSRE, prediction)
                #if (eps_count == 100):
                #    from IPython import embed; embed()
                #    exit()

                if (eps_count % 1000 == 0):
                    print(eps_predictions)

                eps_count += 1
                eps_rewards = []
                eps_predictions = []

                # reset network state
                # wont trigger any bounded units with this input
                #TODO this is currently only for a single layer
                model.set_input_values(np.ones_like(obs) * -np.inf)
                #model.set_input_values(np.zeros_like(obs)) #for some reason behave better at start
                model.step()
                model.introduce_targets(model.read_output_values(), gamma, lmbda)
                model.reset_trace()
        env.close()

