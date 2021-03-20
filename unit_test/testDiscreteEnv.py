from src.gym_sp.supply_chain import *
from src.gym_sp.discrete_supply_chain import *
import json

env = DiscreteSupplyChain()
print(env.discrete_continuous)
print('init state', env.state)
action = env.action_space.sample()
print('action', action)
result = env.step(action)
print('result', result)
state = env.reset()
print('reset state', state)
