import numpy as np
import gym

from src.agents.threshold_policy import ThresholdAgent
from src.gym_sp.supply_chain import SupplyChain

from tensorboardX import SummaryWriter

import json

TEST_EPISODES = 15000
ENV_CONFIG_PATH = "../env_config/envConfig.json"

def create_env ():
    with open(ENV_CONFIG_PATH, 'r') as file:
        config = json.load(file)
    
    env = SupplyChain(config)

    return env

def create_agent (env: SupplyChain):
    action_dim = env.action_space.shape[0]
    num_storages = env.storage_capacity.shape[0]

    eps = []
    Q = []
    for i in env.storage_capacity:
        val = np.ceil(i/5)
        eps.append(val)
        Q.append(i - val)
    Q[0] = env.production_capacity

    agent = ThresholdAgent(tuple(eps), tuple(Q), action_dim)

    return agent

def simulate_episode (env: SupplyChain, agent: ThresholdAgent):
    state = env.reset()
    num_storages = env.storage_capacity.shape[0]
    total_reward = 0
    steps = 0
    while True:
        agent.set_production_level(state, num_storages)
        action = agent.get_action(state)
        new_state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1
        if done:
            break 
        state = new_state

    mean_reward = total_reward / steps
    return total_reward, mean_reward

if __name__ == "__main__":
    env = create_env()
    agent = create_agent(env)
    writer = SummaryWriter(comment="-threshold_policy-v0")
    iter_no = 0

    while True:
        iter_no += 1
        reward_s, reward_m = simulate_episode(env, agent)
        print("%d: total_reward %.3f, mean reward %.3f"%(iter_no, reward_s, reward_m))
        writer.add_scalar("total_reward", reward_s, iter_no)
        writer.add_scalar("mean_reward", reward_m, iter_no)

        if iter_no >= TEST_EPISODES:
            break
    
    writer.close()
    
    