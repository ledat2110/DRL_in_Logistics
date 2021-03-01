from src.gym_sp.supply_chain import *
import json


with open('../env_config/envConfig.json', 'r') as f:
    config = json.load(f)

env = SupplyChain(config)
print('init state', env.state)
print('feasible_action', env.feasible_action())
action = env.action_space.sample()
# action = env.feasible_action()
print('action', action)
demand = env._demand(0)
print('demand', demand)
result = env.step(action)
print('result', result)
print('historical demand', env.historical_demand)
print('feasible action', env.feasible_action())
state = env.reset()
print('reset state', state)
