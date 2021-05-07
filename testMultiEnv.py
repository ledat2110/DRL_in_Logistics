from lib import envs, common
import numpy as np

storage_cost = np.array([0, 0, 0, 1, 2, 3, 1, 2, 3, 1], dtype=np.float32)
truck_cost = np.array([1, 2, 3, 1, 2, 3,1], dtype=np.float32)
storage_cap = np.array([50, 50, 50, 10, 10, 10, 10, 10, 10, 10], dtype=np.int)
link_map = np.array([[1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1]], dtype=np.bool)
env = envs.multi_supply_chain.MultiSupplyChain(7, 3, 2, 2, 4, storage_cost, truck_cost, storage_cap, link_map, 3, 4, 5)
env = common.make_mul_sp()
state = env.reset()
print("state", state)
for i in range(5):
    print(i)
    action = env.action_space.sample()
    print("action", action)
    state, reward, done, info = env.step(action)
    print("new state", state, state.shape)
    print("reward", reward)
    print("done", done)
    print("info", info)
