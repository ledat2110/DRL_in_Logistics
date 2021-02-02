import gym
from gym import error, spaces, utils
from gym.utils import seeding

from src.gym_sp.factory import *
from src.gym_sp.warehouse import *

from typing import Tuple, List, Dict

class SupplyChain(gym.Env):
    def __init__(self, config: Dict):
        self.factory = Factory(config['factory'])

        self.warehouses = []
        warehouses_cf = config['warehouses']
        for warehouse_cf in warehouses_cf:
            warehouse = Warehouse(warehouse_cf)
            self.warehouses.append(warehouse)

        self.action_space = gym.spaces.Discrete(len(self.warehouses) + 1)

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human', close=False):
        raise NotImplementedError
