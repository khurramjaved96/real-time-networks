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
        super(SarsaContinuousPredictionAgent, self).__init__()
        self.expert_agent = expert_agent

    def train(self, env, model, timesteps, epsilon, gamma, lmbda, logger, args):
        """Train the agent
        Args:
            _
        Return:
            -
        """
        rewards_vec = deque(maxlen=1000)
        predictions_vec = deque(maxlen=1000)
        error_trace = 0;
        obs = env.reset()
        for t in range(timesteps):
            self.timestep = t
            model.set_input_values(obs)
            model.step()
            prediction = model.read_output_values()

            # expert_agent does not take the binned input
            action = self.expert_agent.predict(env.unwrapped_obs)

            next_obs, reward, done, info = env.step(action)
            new_target = reward + gamma * model.forward_pass_without_side_effects(next_obs)[0]

            err = model.introduce_targets([new_target], gamma, lmbda)
            error_trace = 0.99 * error_trace + 0.01 * err
            if args.use_imprinting and (err - error_trace > args.imprinting_err_thresh or random.random() < args.imprinting_random_prob):
                #print(f"imprinting now trace: {error_trace} err: {err} t: {t}")
                if args.imprinting_mode == "random":
                    model.imprint_randomly()
                elif args.imprinting_mode == "optical_flow":
                    model.imprint_using_optical_flow()
                elif args.imprinting_mode == "optical_flow_old":
                    model.imprint_using_optical_flow_old()
                else:
                    raise ValueError
            logger.log_imprinting_activity(self.episode, t)
            logger.log_linear_feature_activity(self.episode, t)

            obs = next_obs
            rewards_vec.append(reward)
            predictions_vec.append(prediction[0])
            if done:
                self.episode += 1
            logger.log_step_metrics(self.episode, t)

            if t % 1000 == 0:
                self.MSRE, return_error, return_target = compute_return_error(list(rewards_vec), list(predictions_vec), gamma)
                if self.running_MSRE == -1:
                    self.running_MSRE = self.MSRE
                else:
                    self.running_MSRE = 0.75 * self.running_MSRE + 0.25 * self.MSRE
                logger.log_eps_metrics(self.episode, t, self.MSRE, self.running_MSRE, error_trace, list(predictions_vec), return_target, return_error)
                model.collect_garbage()
        env.close()
