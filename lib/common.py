import torch
import gym

import drl
import numpy as np
import math
import collections
import matplotlib.pyplot as plt

import torch.distributions as D

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
            # action = env.clipping_action(action)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    
    return rewards / count, steps / count

def cal_log_prob (mu_v, logstd_v, actions_v):
    t = D.Independent(D.Normal(mu_v, torch.exp(logstd_v)), 1)
    log_prob = t.log_prob(actions_v)

    return log_prob

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

def plot_fig (y, x, x_labels='steps', y_labels='reward', title='train_reward', legends = ['mean', 'max', 'median'], stepx=5, stepy=5):
    means = []
    maxs = []
    medians = []
    # x = np.array(x) / 52
    # x = x.astype(np.int)
    for i in range(1, len(y) + 1):
        t = y[:i]
        means.append(np.mean(y[:i][-1000:]))
        maxs.append(max(y[:i][-1000:]))
        medians.append(np.median(y[:i][-1000:]))
    plt.figure(figsize=(8, 4))
    plt.plot(x, means)
    plt.plot(x, maxs)
    plt.plot(x, medians)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.yticks(np.arange(np.floor(min(y)), np.ceil(max(y)) + 1, stepy))
    plt.xticks(np.arange(int(min(x)), int(max(x)) +1, stepx))
    plt.legend(legends)
    plt.tight_layout()
    plt.show()
