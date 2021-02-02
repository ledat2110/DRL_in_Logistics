from src.gym_sp.supply_chain import *
import json


with open('./env_config/envConfig.json', 'r') as f:
    config = json.load(f)

env = SupplyChain(config)
print(env.action_space.n)
