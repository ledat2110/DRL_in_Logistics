import gym
import numpy as np

from src.gym_sp.supply_chain import SupplyChain
from src.agents.threshold_policy import ThresholdAgent
env = SupplyChain()
action_dim = env.action_space.shape[0]
num_storages = env.storage_capacity.shape[0]
eps = []
Q = []
for i in env.storage_capacity:
    val = np.ceil(i/10)
    eps.append(val)
    Q.append(i - val)

Q[0] = env.production_capacity
eps = tuple(eps)
Q = tuple(Q)
print(Q, eps)
agent = ThresholdAgent(eps, Q, action_dim)
state = env.reset()
for i in range(4):
    print("env state: ", state)
    agent.set_production_level(state, num_storages)
    action = agent.get_action(state)
    print("action: ", action)
    state, reward, done, _ = env.step(action)
    print(state, reward, done)