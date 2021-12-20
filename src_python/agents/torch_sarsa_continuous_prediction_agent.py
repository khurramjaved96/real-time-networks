import sys
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from ..utils.utils import compute_return_error


class TorchSarsaContinuousPredictionAgent(BaseAgent):
    """Implements the sarsa prediction agent"""

    def __init__(self, expert_agent):
        super(TorchSarsaContinuousPredictionAgent, self).__init__()
        self.expert_agent = expert_agent
        self.loss = nn.MSELoss()

        if torch.cuda.is_available():
            self.device = torch.device('cuda:' + '0')
            print("Using gpu : %s", 'cuda:' + '0')
        else:
           self.device = torch.device('cpu')

    def train(self, env, model, timesteps, epsilon, gamma, lmbda, logger, args):
        """Train the agent
        Args:
            _
        Return:
            -
        """
        self.model = model.to(self.device)
        self.opt = optim.RMSprop(model.parameters(), lr=model.step_size)

        rewards_vec = deque(maxlen=1000)
        predictions_vec = deque(maxlen=1000)
        error_trace = 0;
        obs = env.reset()
        for t in range(timesteps):
            self.timestep = t
            self.opt.zero_grad()
            prediction = model(torch.FloatTensor(obs).to(self.device))

            # expert_agent does not take the binned input
            action = self.expert_agent.predict(env.unwrapped_obs)

            next_obs, reward, done, info = env.step(action)
            with torch.no_grad():
                new_target = reward[0] + gamma * model(torch.FloatTensor(next_obs).to(self.device))

            err = self.loss(prediction.float(), new_target.float())
            error_trace = 0.99 * error_trace + 0.01 * float(err.detach().cpu().data)
            self.opt.zero_grad()
            err.backward()
            self.opt.step()

            obs = next_obs
            rewards_vec.append(reward)
            predictions_vec.append(prediction[0].detach().item())
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
        env.close()
