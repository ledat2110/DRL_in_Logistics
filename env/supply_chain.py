import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.stats import *
import numpy as np
import math

from typing import Tuple, List, Dict

class SupplyChain (gym.Env):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=1, max_production: int=3,
                    store_cost: np.array=np.array([0, 2, 0, 0], dtype=np.float32),
                    truck_cost: np.array=np.array([3, 3, 0], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 10, 10], dtype=np.float32),
                    penalty_cost: int=1, price: int=2.5, max_demand: int=3, num_period: int=25, periodic_demand: bool=True):
        self.num_period = num_period
        self.unit_cost = production_cost
        self.production_capacity = max_production
        self.storage_capacity = storage_capacity
        self.product_price = price
        self.storage_cost = store_cost
        self.penalty_cost = penalty_cost
        self.truck_cost = truck_cost
        self.truck_capacity = cap_truck
        self.demand_max = max_demand
        self.num_stores = n_stores
        self.periodic_demand = periodic_demand

        self.action_dim = self.storage_capacity.shape[0]
        self.action_space = gym.spaces.Box(
                low = np.zeros(self.action_dim),
                high = self.storage_capacity,
                dtype = np.float32
                )

        self.demand_dim = self.num_stores
        self.state_dim = self.storage_capacity.shape[0] + self.demand_dim * 2
        self.observation_space = gym.spaces.Box(
                low = np.full(self.state_dim, -np.inf),
                high = np.full(self.state_dim, np.inf),
                dtype = np.float32
                )

        self._init()

    def _init (self):
        self.period = 0
        self.demand = np.zeros(self.num_stores, dtype=int)
        self._update_demand()
        self.old_demand = self.demand.copy()
        self.inventory = self.storage_capacity.copy() / 2
        self._update_state()

    def step (self, action: np.ndarray):
        # cliping action into feasible action
        action = self._clipping_action(action)
        # update inventory
        self._update_inventory(action)

        # update reward
        reward = self._update_reward(action)
        info = "Demand was: ", self.demand

        # update state
        self._update_state()

        # update historical demand
        self.old_demand = self.demand.copy()

        # update t
        self.period += 1

        # update demand
        self._update_demand()

        # set done
        done = False
        if self.period >= self.num_period:
            done = True

        return self.state, reward, done, info

    def _clipping_action (self, action: np.ndarray) -> np.ndarray:
        upper_bound = self.storage_capacity - self.inventory
        upper_bound[0] = upper_bound[0] + np.sum(action[1:])
        action = np.clip(action, np.zeros(self.action_dim), upper_bound)

        if np.sum(action[1:]) > self.inventory[0]:
            action[1:] = action[1:] * self.inventory[0] / np.sum(action[1:])
        action = np.around(action, 4)
        action = np.array(action, dtype=np.float32, copy=False)

        return action

    def _update_inventory (self, action: np.ndarray):
        self.inventory[0] = min(self.inventory[0] + action[0] - np.sum(action[1:]), self.storage_capacity[0])
        self.inventory[0] = max(0, self.inventory[0])
        self.inventory[1:] = np.minimum(self.inventory[1:] + action[1:] - self.demand, self.storage_capacity[1:])
        self.inventory = np.around(self.inventory, 4)
        self.inventory = np.array(self.inventory, copy=False, dtype=np.float32)

    def _update_reward (self, action: np.ndarray) -> float:
        zeros_array = np.zeros_like(self.inventory)

        revenue = np.sum(self.demand * self.product_price)
        production_cost = self.unit_cost * action[0]
        storage_cost = np.sum(np.maximum(zeros_array, self.inventory) * self.storage_cost)
        penalty_cost = np.sum(np.minimum(zeros_array, self.inventory) * self.penalty_cost)
        truck_cost = np.sum(np.ceil(action[1:]/self.truck_capacity) * self.truck_cost)
        
        reward = revenue - production_cost - storage_cost + penalty_cost - truck_cost
        return reward

    def _update_state (self):
        inventory = self.inventory.copy()
        demand = self.demand.copy()
        old_demand = self.old_demand.copy()
        self.state = np.concatenate((inventory, demand, old_demand))
        self.state = np.array(self.state, copy=False, dtype=np.float32)

    def _update_demand (self):
        demand = np.zeros(self.num_stores, dtype=int)
        for i in range(self.num_stores):
            if self.periodic_demand == True:
                eps = np.random.choice([0, 1], p=(0.5, 0.5))
                rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
                val = .5 * self.demand_max * np.sin(rad) + .5 * self.demand_max + eps
                demand[i] = int(np.floor(val))
            else:
                demand[i] = np.random.randint(low=0, high=self.demand_max)

        self.demand = demand

    def reset (self) -> np.ndarray:
        self._init()
        return self.state

    def render (self, mode='human', close=False):
        raise NotImplementedError

class DiscreteSupplyChain (gym.Env):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=1, max_production: int=3,
                    store_cost: np.array=np.array([0, 2, 0, 0], dtype=np.float32),
                    truck_cost: np.array=np.array([3, 3, 0], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 10, 10], dtype=np.float32),
                    penalty_cost: int=1, price: int=2.5, max_demand: int=3, num_period: int=48, periodic_demand: bool=True):
        self.num_period = num_period
        self.unit_cost = production_cost
        self.production_capacity = max_production
        self.storage_capacity = storage_capacity
        self.product_price = price
        self.storage_cost = store_cost
        self.penalty_cost = penalty_cost
        self.truck_cost = truck_cost
        self.truck_capacity = cap_truck
        self.demand_max = max_demand
        self.num_stores = n_stores
        self.periodic_demand = periodic_demand
        self.actions_per_store = 4

        self.action_dim = self.storage_capacity.shape[0]
        available_actions = np.zeros((self.action_dim, self.actions_per_store))
        available_actions[0] = [0, self.production_capacity, self.production_capacity * 2, self.production_capacity * 4]
        for i in range(self.num_stores + 1):
            available_actions[i:, :] = [0, self.storage_capacity[i] / 2, self.storage_capacity[i], self.storage_capacity[i] * 2]
        self.available_actions = available_actions

        continuous_action_shape = (self.actions_per_store ** (self.num_stores + 1), self.num_stores + 1)
        self.discrete_continuous = np.zeros(continuous_action_shape)
        for i in range(self.num_stores + 1):
            step = self.actions_per_store ** (self.num_stores - i)
            for j in range(0, continuous_action_shape[0], step):
                idx = int(j / step) % self.actions_per_store
                self.discrete_continuous[j:j+step, i] = available_actions[i, idx]
                        
        self.action_space = spaces.Discrete(len(self.discrete_continuous))

        self.demand_dim = self.num_stores
        self.state_dim = self.storage_capacity.shape[0] + self.demand_dim * 2
        self.observation_space = gym.spaces.Box(
                low = np.full(self.state_dim, -np.inf),
                high = np.full(self.state_dim, np.inf),
                dtype = np.float32
                )

        self._init()

    def _init (self):
        self.period = 0
        self.demand = np.zeros(self.num_stores, dtype=int)
        self._update_demand()
        self.old_demand = self.demand.copy()
        self.inventory = self.storage_capacity.copy() / 2
        self._update_state()

    def step (self, action: int):
        # cliping action into feasible action
        action = self.discrete_continuous[action].copy()

        # update inventory
        self._update_inventory(action)

        # update reward
        reward = self._update_reward(action)
        info = "Demand was: ", self.demand

        # update state
        self._update_state()

        # update historical demand
        self.old_demand = self.demand.copy()

        # update t
        self.period += 1

        # update demand
        self._update_demand()

        # set done
        done = False
        if self.period >= self.num_period:
            done = True

        return self.state, reward, done, info

    def _clipping_action (self, action: np.ndarray) -> np.ndarray:
        upper_bound = self.storage_capacity - self.inventory
        upper_bound[0] = upper_bound[0] + np.sum(action[1:])
        action = np.clip(action, np.zeros(self.action_dim), upper_bound)

        if np.sum(action[1:]) > self.inventory[0]:
            action[1:] = action[1:] * self.inventory[0] / np.sum(action[1:])
        action = np.around(action, 4)

        return action

    def _update_inventory (self, action: np.ndarray):
        self.inventory[0] = min(self.inventory[0] + action[0] - np.sum(action[1:]), self.storage_capacity[0])
        self.inventory[0] = max(0, self.inventory[0])
        self.inventory[1:] = np.minimum(self.inventory[1:] + action[1:] - self.demand, self.storage_capacity[1:])
        self.inventory = np.around(self.inventory, 4)

    def _update_reward (self, action: np.ndarray) -> float:
        zeros_array = np.zeros_like(self.inventory)

        revenue = np.sum(self.demand * self.product_price)
        production_cost = self.unit_cost * action[0]
        storage_cost = np.sum(np.maximum(zeros_array, self.inventory) * self.storage_cost)
        penalty_cost = np.sum(np.minimum(zeros_array, self.inventory) * self.penalty_cost)
        truck_cost = np.sum(np.ceil(action[1:]/self.truck_capacity) * self.truck_cost)
        
        reward = revenue - production_cost - storage_cost + penalty_cost - truck_cost
        return reward

    def _update_state (self):
        inventory = self.inventory.copy()
        demand = self.demand.copy()
        old_demand = self.old_demand.copy()
        self.state = np.concatenate((inventory, demand, old_demand))

    def _update_demand (self):
        demand = np.zeros(self.num_stores, dtype=int)
        for i in range(self.num_stores):
            if self.periodic_demand == True:
                eps = np.random.choice([0, 1], p=(0.5, 0.5))
                rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
                val = .5 * self.demand_max * np.sin(rad) + .5 * self.demand_max + eps
                demand[i] = int(np.floor(val))
            else:
                demand[i] = np.random.randint(low=0, high=self.demand_max)

        self.demand = demand

    def reset (self) -> np.ndarray:
        self._init()
        return self.state

    def render (self, mode='human', close=False):
        raise NotImplementedError
