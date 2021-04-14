import numpy as np
import gym
import collections

import drl_lib
from drl_lib.env.supply_chain import SupplyChain

from tensorboardX import SummaryWriter

TEST_EPISODES = 3000
ENV_CONFIG_PATH = "../env_config/envConfig.json"

Result = collections.namedtuple("Result", field_names=['reward', 'step', 'done'])


def create_agent (env: SupplyChain):
    action_dim = env.action_space.shape[0]

    eps = env.storage_capacity.copy() / 10
    Q = env.storage_capacity.copy()

    agent = drl_lib.agent.ThresholdAgent(tuple(eps), tuple(Q), action_dim)

    return agent

def simulate_episode (env: SupplyChain, agent):
    state = env.reset()
    num_storages = env.num_stores
    total_reward = 0
    steps = 0
    while True:
        agent.set_production_level(state, num_storages)
        action = agent.get_action(state)
        new_state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1
        state = new_state
        yield Result(total_reward, steps, done)
        if done:
            env.reset()
            total_reward = 0
            steps = 0

if __name__ == "__main__":
    #env = SupplyChain(num_period=25, store_cost=np.array([0, 2, 2, 2]), truck_cost=np.array([3, 3, 3]), price=7)
    env = SupplyChain(periodic_demand=False)
    agent = create_agent(env)
    writer = SummaryWriter(comment="-threshold_policy-v0")
    iter_no = 0
    total_reward = []
    done_episode = 0

    with drl_lib.tracker.RewardTracker(writer, 100) as tracker:
        for step_idx, result in enumerate(simulate_episode(env, agent)):
            reward, step, done = result.reward, result.step, result.done
            if done:
                tracker.update(reward, step_idx)
                done_episode += 1

            if done_episode > TEST_EPISODES:
                break
        
    writer.close()
    
    