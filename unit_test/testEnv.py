import json
import drl_lib

env = drl_lib.env.supply_chain.SupplyChain()
print('init state', env.state)
action = env.action_space.sample()
print('action', action)
result = env.step(action)
print('result', result)
state = env.reset()
print('reset state', state)

env = drl_lib.env.supply_distribution10.SupplyDistribution()
print('init state', env.state)
action = env.action_space.sample()
print('action', action)
result = env.step(action)
print('result', result)
state = env.reset()
print('reset state', state)
