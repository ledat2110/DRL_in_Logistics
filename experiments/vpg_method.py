import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import drl_lib
import gym
import time
import math

from tensorboardX import SummaryWriter

GAMMA = 1
LEARNING_RATE = 5e-3
ENTROPY_WEIGHT = 0.01
BATCH_SIZE = 64

REWARD_STEPS = 1
GRAD_L2_CLIP = 0.1

class LogisticsPGN (nn.Module):
    def __init__ (self, ob_dim, action_dim):
        super(LogisticsPGN, self).__init__ ()

        self.base = nn.Sequential(
                nn.Linear(ob_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
                )

        self.mu = nn.Sequential(
                nn.Linear(64, action_dim),
                )

        w = torch.zeros(action_dim)
        self.log_std = nn.Parameter(w)

    def _format (self, x):
        fx = x
        if not isinstance(fx, torch.Tensor):
            fx = torch.Tensor(x, dtype=torch.float32)
            fx = fx.unsqueeze(0)
        fx = fx.float()
        return fx

    def forward (self, x):
        fx = self._format(x)
        out_base = self.base(fx)
        out_mean = self.mu(out_base)

        return out_mean, self.log_std

class Agent (drl_lib.agent.BaseAgent):
    def __init__ (self, model: nn.Module, device="cpu", preprocessor=drl_lib.utils.Preprocessor.default_tensor):
        super(Agent, self).__init__()
        self.model = model
        self.device = device
        self.preprocessor = preprocessor

    def __call__ (self, state: np.ndarray):
        if self.preprocessor is not None:
            state = self.preprocessor(state)
        if torch.is_tensor(state):
            state = state.to(self.device)

        mu_v, var_v = self.model(state)
        mu = mu_v.squeeze(0).data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()

        action = np.random.normal(mu, sigma)
        return action

def calc_logprob (mu_v: torch.Tensor, var_v: torch.Tensor, action_v: torch.Tensor) -> torch.Tensor:
    p1 = - ((mu_v - action_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))

    return p1 + p2


