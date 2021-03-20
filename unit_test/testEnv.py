from src.gym_sp.supply_chain import *
import json

env = SupplyChain()
print('init state', env.state)
action = env.action_space.sample()
print('action', action)
result = env.step(action)
print('result', result)
state = env.reset()
print('reset state', state)
