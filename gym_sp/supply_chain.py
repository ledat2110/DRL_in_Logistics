import gym
from gym import error, spaces, utils
from gym.utils import seeding

from factory import *
from warehouse import *

from typing import Tuple, List, Dict

class SupplyChain(gym.Env):
    def __init__(self, config: Dict):
        self.config = config
    def step(self, action):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def render(self, mode='human', close=False):
        raise NotImplementedError
