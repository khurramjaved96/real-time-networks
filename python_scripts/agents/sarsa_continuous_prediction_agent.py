import sys
import random
import numpy as np
from collections import deque

from .base_agent import BaseAgent

# sys.path.append("../utils")
# from utils import compute_return_error
from ..utils.utils import compute_return_error


class SarsaContinuousPredictionAgent(BaseAgent):
    """Implements the sarsa prediction agent"""

    def __init__(self, expert_agent):
        self.expert_agent = expert_agent

    def train(self, env, model, timesteps, epsilon, gamma, lmbda, logger):
        """Train the agent
        Args:
            _
        Return:
            -
        """
        rewards_vec = deque(maxlen=1000)
        predictions_vec = deque(maxlen=1000)
        running_MSRE = -1
        obs = env.reset()
        for t in range(timesteps):

            from IPython import embed; embed()
            model.set_input_values(obs)
            model.step()
            prediction = model.read_output_values()

            action = self.expert_agent.predict(env.unwrapped_obs)

            next_obs, reward, done, info = env.step(action)
            bootstrap_prediction = model.forward_pass_without_side_effects(next_obs)
            new_target = reward + gamma * bootstrap_prediction[0]

            err = model.introduce_targets([new_target], gamma, lmbda)
            obs = next_obs

            rewards_vec.append(reward)
            predictions_vec.append(prediction[0])
            logger.log_step_metrics(eps_count, t)

            if t % 1000 == 0:
                MSRE, return_error, return_target = compute_return_error(list(rewards_vec), list(predictions_vec), gamma)
                if t < 1001:
                    running_MSRE = MSRE
                else:
                    running_MSRE = 0.99 * running_MSRE + 0.01 * MSRE
                logger.log_eps_metrics(t, t, MSRE, running_MSRE, err, list(predictions_vec), return_target, return_error)
        env.close()
