import torch
import gym

import drl
import numpy as np
import math
import collections

from . import envs

def test_net (net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = drl.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = env.clipping_action(action)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    
    return rewards / count, steps / count

def cal_log_prob (mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))

    return p1 + p2

def make_mul_sp ():
    store_cost = np.array([0, 0, 0, 1, 2, 2, 0, 1, 2, 1], dtype=np.float32)
    truck_cost = np.array([3, 3, 3, 3, 0, 3, 3], dtype=np.float32)
    storage_cap = np.array([50, 50, 50, 10, 10, 10, 10, 10, 10, 10], dtype=np.int)
    link_map = np.array([[1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]], dtype=np.bool)
    env = envs.multi_supply_chain.MultiSupplyChain(7, 3, 2, 1, 3, store_cost, truck_cost, storage_cap, link_map, 1, 2.5, 3, 25, True)
    return env

class PseudoCountRewardWrapper (gym.Wrapper):
    def __init__ (self, env, hash_function, reward_scale: float=1.0):
        super(PseudoCountRewardWrapper, self).__init__(env)
        self.hash_function = hash_function
        self.reward_scale = reward_scale
        self.counts = collections.Counter()

    def _count_observation (self, obs) -> float:
        h = self.hash_function(obs)
        self.counts[h] += 1
        return np.sqrt(1 / self.counts[h])

    def step (self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self._count_observation(obs)
        return obs, reward + self.reward_scale * extra_reward, done, info

def counts_hash (obs):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))
