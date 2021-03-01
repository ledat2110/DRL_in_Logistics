import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.stats import *
import numpy as np
import math

from typing import Tuple, List, Dict

class SupplyChain(gym.Env):
    def __init__(self, config: Dict):
        self.num_period = config['num_period']
        self.unit_cost = config['unit_cost']
        self.production_capacity = config['production_capacity']
        self.init_inventory = np.array(config['init_inventory'], dtype = np.int32)
        self.storage_capacity = np.array(config['storage_capacity'], dtype=np.int32)
        self.product_price = np.array(config['product_price'], dtype = np.int32)
        self.storage_cost = np.array(config['storage_cost'], dtype=np.int32)
        self.penalty_cost = np.array(config['penalty_cost'], dtype=np.int32)
        self.truck_cost = np.array(config['truck_cost'], dtype=np.int32)
        self.truck_capacity = np.array(config['truck_capacity'], dtype=np.int32)
        self.seed_int = config['seed']
        self.dist_type = config['distribution']
        self.demand_max = config['demand_max']
        self.num_stores = config['num_stores']

        self.seed(self.seed_int)
        self.action_dim = self.storage_capacity.shape[0]
        self.action_max = [self.production_capacity] + config['storage_capacity'][1:]
        self.action_space = gym.spaces.Box(
                low = np.zeros(self.action_dim),
                high = np.zeros(self.action_dim) + self.action_max,
                dtype = np.int32
                )

        self.demand_dim = self.storage_capacity.shape[0]
        self.state_dim = self.storage_capacity.shape[0] + (self.demand_dim - 1) * 2
        self.observation_space = gym.spaces.Box(
                low = -np.ones(self.state_dim) * self.num_period * 10 * self.storage_capacity.max(),
                high = np.ones(self.state_dim) * self.num_period * self.storage_capacity.max(),
                dtype = np.int32
                )

        self.distribution = {
                'poisson': [poisson, {'mu': self.storage_capacity}],
                'binom': [binom, {'n': self.storage_capacity, 'p': np.random.rand(self.demand_dim)}],
                'randint': [randint, {'low': np.zeros_like(self.storage_capacity), 'high': self.storage_capacity}]
                }

        self._init()

    def seed(self, seed_int = None):
        if seed_int != None:
            np.random.seed(seed_int)

    def step(self, action: np.ndarray):
        # feasible_action = self.feasible_action()
        assert action.shape[0] == self.action_space.shape[0]
        assert self.check_feasible_action(action) == True
        # assert (action <= feasible_action).all() == True
        # assert self.inventory[0] >= sum(action[1:])

        unit_cost = self._unit_cost(action[0])
        truck_cost = self._transportation_cost(action[1:])

        self.inventory += action
        self.inventory[0] -= action[1:].sum()
        self.inventory[0] = min(self.inventory[0], self.storage_capacity[0])

        demand = self._demand()
        self.historical_demand.append(demand)
        revenue = self._revenue(demand)
        for i in range(1, len(self.inventory)):
            self.inventory[i] -= demand[i]
            self.inventory[i] = min(self.inventory[i], self.storage_capacity[i])

        storage_cost = self._storage_cost()
        penalty_cost = self._penalty_cost()

        reward = revenue - unit_cost - truck_cost - storage_cost + penalty_cost

        self.period += 1
        if self.period >= self.num_period:
            done = True
        else:
            done = False

        self._update_state()

        return self.state, reward, done, {}

    def _init(self):
        self.period = 0
        self.historical_demand = [np.zeros(self.demand_dim), np.zeros(self.demand_dim)]
        self.inventory = self.init_inventory.copy()
        self._update_state()

    def _update_state(self) -> np.ndarray:
        self.state = np.concatenate((self.inventory, self.historical_demand[-1][1:], self.historical_demand[-2][1:]))

    def _demand(self) -> np.ndarray:
        demand = [0]
        for i in range(self.num_stores):
            eps = np.random.choice([0, 1], p=(0.5, 0.5))
            rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
            val = .5 * self.demand_max * np.sin(rad) + .5 * self.demand_max + eps
            demand.append(int(np.floor(val)))

        # dist = self.distribution[self.dist_type]
        # param = dist[1]
        # dist = dist[0]

        # demand = dist.rvs(**param)
        demand = np.asarray(demand)
        return demand

    def _unit_cost(self, products: int) -> int:
        return self.unit_cost * products

    def _storage_cost(self) -> int:
        cost = 0
        for i, inv in enumerate(self.inventory):
            cost += self.storage_cost[i] * max(inv, 0)

        return cost

    def _transportation_cost(self, delivered_products: List) -> int:
        cost = 0
        for i in range(len(delivered_products)):
            cost += self.truck_cost[i] * math.ceil(delivered_products[i] / self.truck_capacity[i])

        return cost

    def _penalty_cost(self) -> int:
        cost = 0
        for i, inv in enumerate(self.inventory):
            cost += self.penalty_cost[i] * min(inv, 0)

        return cost

    def _revenue(self, demand: np.ndarray) -> int:
        revenue = 0
        for i in range(1, self.inventory.shape[0]):
            revenue += self.product_price[i] * min(self.inventory[i], demand[i])

        return revenue

    def feasible_action(self) -> np.ndarray:
        action = [self.production_capacity]
        for i in range(1, self.inventory.shape[0]):
            action.append(self.storage_capacity[i] - self.inventory[i])

        return np.asarray(action, dtype=np.int32)

    def check_feasible_action(self, action: np.ndarray) -> bool:
        result = (action[0] <= self.production_capacity) & (np.sum(action[1:]) <= self.inventory[0]).all()

        return result

    def reset(self) -> np.ndarray:
        self._init()
        return self.state

    def render(self, mode='human', close=False):
        raise NotImplementedError
