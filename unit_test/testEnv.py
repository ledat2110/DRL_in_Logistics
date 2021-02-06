from src.gym_sp.supply_chain import *
import json


with open('../env_config/envConfig.json', 'r') as f:
    config = json.load(f)

env = SupplyChain(config)
print('init state', env.state)
action = np.ones(env.action_space.shape, dtype=np.int32) * 3
print('action', action)
result = env.step(action)
print('result', result)
print('historical demand', env.historical_demand)
print('feasible action', env.feasible_action())
state = env.reset()
print('reset state', state)
print('feasible_action', env.feasible_action())
