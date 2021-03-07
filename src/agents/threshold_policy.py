import gym
import numpy as np
from typing import List, Tuple

class ThresholdAgent:
    def __init__ (self, eps: np.array, Q: np.array, action_dim: int):
        self.eps = eps
        self.Q = Q
        self.action_dim = action_dim
        self.production_flag = True

    def get_action (self, state: np.ndarray):
        action = np.zeros(self.action_dim, dtype=np.int32)
        action[0] = self.Q[0] if self.production_flag else 0

        for i in range(1, self.action_dim):
            if state[i] < self.eps[i]:
                action[i] = self.Q[i]

        return action

    def set_production_level (self, state: np.ndarray, num_storages: int):
        self.production_flag = False
        if (state[0] - np.sum(state[1:num_storages+1])) < self.eps[0]:
            self.production_flag = True
