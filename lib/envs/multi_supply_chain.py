import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.stats import *
import numpy as np
import math
import cv2

from typing import Tuple, List, Dict

class MultiSupplyChain (gym.Env):
    def __init__ (self, n_stores: int, n_warehouse: int,
            cap_truck: int, production_cost: int, max_production: int,
            store_cost: np.array, truck_cost: np.array, storage_cap: np.array, link_map: np.array, 
            penalty_cost: float, price: float, max_demand: float,
            num_period: int=25, periodic_demand: bool=True):

        self.n_stores = n_stores
        self.n_warehouse = n_warehouse
        self.cap_truck = cap_truck
        self.production_cost = production_cost
        self.max_production = max_production
        self.store_cost = store_cost
        self.truck_cost = truck_cost
        self.storage_cap = storage_cap
        self.link_map = link_map
        self.penalty_cost = penalty_cost
        self.price = price
        self.max_demand = max_demand
        self.num_period = num_period
        self.periodic_demand = periodic_demand

        self.action_dim = self.n_warehouse + self.n_stores
        self.action_space = gym.spaces.Box(
                low=0,
                high=self.storage_cap,
                dtype=np.float32)

        self.demand_dim = self.n_stores
        self.state_shape = (4, self.n_warehouse, self.n_stores)
        self.observation_space = gym.spaces.Box(
                low = np.full((4, 84, 84), -np.inf),
                high = np.full((4, 84, 84), np.inf),
                dtype = np.float32
                )

        self.reset()

    def reset (self):
        self.period = 0
        self.demand = np.zeros(self.n_stores, dtype=int)
        self._update_demand()
        self.old_demand = self.demand.copy()
        self.inventory = self.storage_cap.copy() / 2
        state = self._update_state()

        return state

    def step (self, action: np.array):
        # clipping action into feasible action
        action = self._clipping_action(action)

        # update inventory
        self._update_inventory(action)

        # update reward
        reward = self._update_reward(action)
        info = "Demand was: ", self.demand

        # update state
        state = self._update_state()

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

        return state, reward, done, info

    def _clipping_action (self, action: np.array):
        upper_bound = self.storage_cap - self.inventory
        #upper_bound[:self.n_warehouse] += np.sum(action[self.n_warehouse:])
        action = np.clip(action, np.zeros_like(action), upper_bound)

        delivery = action[self.n_warehouse:]
        for i in range(self.n_warehouse):
            stores = self.link_map[i]
            if np.sum(delivery[stores]) > self.inventory[i]:
                delivery[stores] *= self.inventory[i] / np.sum(delivery[stores])

        action = np.around(action, 4).astype(np.float32)

        return action

    def _update_inventory (self, action: np.array):
        delivery = action[self.n_warehouse:]
        for i in range(self.n_warehouse):
            stores = self.link_map[i]
            self.inventory[i] = min(self.inventory[i] + action[i] - np.sum(delivery[stores]), self.storage_cap[i])
            self.inventory[i] = max(0, self.inventory[i])

        self.inventory[self.n_warehouse:] = np.minimum(self.inventory[self.n_warehouse:] + delivery - self.demand, self.storage_cap[self.n_warehouse:])
        self.inventory = np.around(self.inventory, 4).astype(np.float32)

    def _update_reward (self, action: np.array):
        zeros = np.zeros_like(self.inventory)

        revenue = np.sum(self.demand * self.price)
        production_cost = np.sum(self.production_cost * action[:self.n_warehouse])
        storage_cost = np.sum(np.maximum(zeros, self.inventory) * self.store_cost)
        penalty_cost = np.sum(np.minimum(zeros, self.inventory) * self.penalty_cost)
        truck_cost = np.sum(np.ceil(action[self.n_warehouse:] / self.cap_truck) * self.truck_cost)

        reward = revenue - production_cost - storage_cost + penalty_cost - truck_cost
        return reward

    def _update_state (self):
        inventory = self.inventory.copy()
        demand = self.demand.copy()
        old_demand = self.old_demand.copy()
        state = np.zeros(self.state_shape, dtype=np.float32)

        for i in range(self.n_warehouse):
            stores = self.link_map[i]
            state[0, i, :][stores] = inventory[self.n_warehouse:][stores]
            state[1, i, :][stores] = inventory[self.n_warehouse:][stores]
            state[2, i, :][stores] = demand[stores]
            state[3, i, :][stores] = old_demand[stores]

        s = []
        for i in range(4):
            img = state[i]
            resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
            x_t = resized_img[18:102, :]
            x_t = np.reshape(x_t, [1, 84, 84])
            s.append(x_t)

        return np.concatenate(s, axis=0).astype(np.float32)

    def _update_demand (self):
        demand = np.zeros(self.demand_dim, dtype=int)
        for i in range(self.n_stores):
            if self.periodic_demand == True:
                eps = np.random.choice([0, 1], p=(0.5, 0.5))
                rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
                val = .5 * self.max_demand * np.sin(rad) + .5 * self.max_demand + eps
                demand[i] = int(np.floor(val))
            else:
                demand[i] = np.random.randint(low=0, high=self.max_demand)

        self.demand = demand

