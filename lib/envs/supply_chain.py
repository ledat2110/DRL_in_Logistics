import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.stats import *
import numpy as np
import math
import pygame
import collections

from PIL import Image
from typing import Tuple, List, Dict

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 40
SPEED = 20

# COLOR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)

Point = collections.namedtuple("Point", ['x', 'y'])

# POS
PLANT_POS = Point(20, 220)
WAREHOUSE_POS = Point(210, 220)
RETAIL_POS = [Point(440, 40), Point(440, 220), Point(440, 400)]
REWARD_POS = [0, 0]
PRODUCTION_TEXT_POS = [(WAREHOUSE_POS.x + PLANT_POS.x + BLOCK_SIZE) / 2, PLANT_POS.y]

# font
FONT_SIZE = 20
pygame.font.init()
font = pygame.font.SysFont("Arial", FONT_SIZE)

class SupplyChain (gym.Env):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=1, max_production: int=3,
                    store_cost: np.array=np.array([0, 2, 2, 2], dtype=np.float32),
                    truck_cost: np.array=np.array([1, 1, 1], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 10, 10], dtype=np.float32),
                    penalty_cost: int=1, price: int=3,
                    max_demand: int=4, num_period: int=52, periodic_demand: bool=True,
                    matrix_state: bool=False, v_demand: int=0, m_demand: int=0,
                    disp: bool=False):
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
        self.v_demand = v_demand
        self.m_demand = m_demand

        self.action_dim = self.storage_capacity.shape[0]
        self.action_space = gym.spaces.Box(
                low = np.zeros(self.action_dim),
                high = self.storage_capacity,
                dtype = np.float32
                )

        self.demand_dim = self.num_stores
        self.matrix_state = matrix_state
        if self.matrix_state == False:
            self.state_dim = self.storage_capacity.shape[0] + self.demand_dim * 2
        else:
            self.state_dim = (3, self.storage_capacity.shape[0])
        self.observation_space = gym.spaces.Box(
                low = np.full(self.state_dim, -np.inf),
                high = np.full(self.state_dim, np.inf),
                dtype = np.float32
                )

        self.reset()
        self.disp = disp
        if self.disp:
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Logistics")
            self.clock = pygame.time.Clock()

    def reset (self):
        self.period = 0
        self.demand = np.zeros(self.num_stores, dtype=int)
        self._update_demand()
        self.old_demand = self.demand.copy()
        self.inventory = self.storage_capacity.copy() / 2
        state = self._update_state()
        self.sum_reward = 0

        return state

    def step (self, action: np.ndarray):
        # cliping action into feasible action
        action = self.clipping_action(action)
        # print(action)
        # update inventory
        self._update_inventory(action)

        # update reward
        reward = self._update_reward(action)
        info = "Demand was: ", self.demand
        self.sum_reward += reward

        # update state
        state = self._update_state()

        # update historical demand
        self.old_demand = self.demand.copy()

        # render
        if self.disp:
            self.render(action)
            self.clock.tick(SPEED)

        # update t
        self.period += 1

        # update demand
        self._update_demand()

        # set done
        done = False
        if self.period >= self.num_period:
            done = True

        # print(state)

        return state, reward, done, info

    def clipping_action (self, action: np.ndarray) -> np.ndarray:
        upper_bound = self.storage_capacity - self.inventory
        action = np.clip(action, np.zeros(self.action_dim), upper_bound)
        if np.sum(action[1:]) > self.inventory[0] and np.sum(action[1:]) != 0:
            action[1:] = action[1:] * self.inventory[0] / np.sum(action[1:])
        action = np.around(action, 4).astype(np.float32)

        return action

    def _update_inventory (self, action: np.ndarray):
        self.inventory[0] = min(self.inventory[0] + action[0] - np.sum(action[1:]), self.storage_capacity[0])
        self.inventory[0] = max(0, self.inventory[0])
        self.inventory[1:] = np.minimum(self.inventory[1:] + action[1:] - self.demand, self.storage_capacity[1:])
        self.inventory = np.around(self.inventory, 4).astype(np.float32)

    def _update_reward (self, action: np.ndarray) -> float:
        zeros_array = np.zeros_like(self.inventory)

        revenue = np.sum(np.maximum(self.demand, self.inventory[1:])  * self.product_price)
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
        if self.matrix_state == False:
            state = np.concatenate((inventory, demand, old_demand)).astype(np.float32)
        else:
            state = []
            state.append(inventory)
            demand = np.concatenate([[0], demand]).astype(np.float32)
            old_demand = np.concatenate([[0], old_demand]).astype(np.float32)
            state.append(demand)
            state.append(old_demand)
            state = np.array(state, dtype=np.float32, copy=False)

        return state

    def _update_demand (self):
        demand = np.zeros(self.num_stores, dtype=int)
        # self.m_demand = np.random.uniform(1/3, 3)
        # self.v_demand = np.random.uniform(1/3, 3)
        for i in range(self.num_stores):
            eps = np.random.uniform(-self.demand_max/2, self.demand_max/2)
            v = self.demand_max * (self.v_demand ** (self.period / self.num_period)) \
                * np.sin(2*(self.period + 2* i) /26) / 4
            m = self.demand_max + self.m_demand * (self.period / self.num_period)
            demand[i] = np.floor(m + eps + v)
            # if self.periodic_demand == True:
            #     eps = np.random.choice([0, 1], p=(0.5, 0.5))
            #     rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
            #     val = .5 * self.demand_max * np.sin(rad) + .5 * self.demand_max + eps
            #     demand[i] = int(np.floor(val))
            # else:
            #     demand[i] = np.random.randint(low=0, high=self.demand_max)

        self.demand = demand

    def get_demand (self):
        self.period = 0
        demands = []
        while self.period < self.num_period:
            self._update_demand()
            demands.append(self.demand)
            self.period += 1
        return demands

    def render (self, action, mode='human', close=False):
        self.display.fill(WHITE)
        pygame.draw.rect(self.display, RED, pygame.Rect(PLANT_POS.x, PLANT_POS.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.line(self.display, BLACK, (PLANT_POS.x + BLOCK_SIZE, PLANT_POS.y+BLOCK_SIZE/2), (WAREHOUSE_POS.x, WAREHOUSE_POS.y+BLOCK_SIZE/2))
        production = font.render("%.3f"%action[0], True, RED)
        self.display.blit(production, PRODUCTION_TEXT_POS)
        pygame.draw.rect(self.display, GREEN, pygame.Rect(WAREHOUSE_POS.x, WAREHOUSE_POS.y, BLOCK_SIZE, BLOCK_SIZE))
        wh_store = font.render("%.3f"%self.inventory[0], True, BLACK)
        self.display.blit(wh_store, (WAREHOUSE_POS.x, WAREHOUSE_POS.y+BLOCK_SIZE))
        for idx, p in enumerate(RETAIL_POS):
            pygame.draw.rect(self.display, BLUE, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.line(self.display, BLACK, (WAREHOUSE_POS.x+BLOCK_SIZE, WAREHOUSE_POS.y+BLOCK_SIZE/2), (p.x, p.y+BLOCK_SIZE/2))
            store = font.render("%.3f"%self.inventory[idx+1], True, BLACK)
            self.display.blit(store, (p.x, p.y+BLOCK_SIZE))
            deliver = font.render("%.3f"%action[idx+1], True, BLACK)
            self.display.blit(deliver, ((WAREHOUSE_POS.x+BLOCK_SIZE+p.x)/2, (p.y+WAREHOUSE_POS.y)/2))
            demand = font.render("%.3f"%self.old_demand[idx], True, GREEN)
            self.display.blit(demand, (p.x + BLOCK_SIZE *2, p.y))
        reward_txt = font.render("Reward: %.3f"%self.sum_reward, True, BLACK)
        self.display.blit(reward_txt, REWARD_POS)
        pygame.display.flip()

    def save_img (self, name):
        data = pygame.image.tostring(self.display, 'RGB')
        image = Image.frombytes('RGB', (WIDTH, HEIGHT), data)
        image.save(name)

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

        self.reset()

    def reset (self):
        self.period = 0
        self.demand = np.zeros(self.num_stores, dtype=int)
        self._update_demand()
        self.old_demand = self.demand.copy()
        self.inventory = self.storage_capacity.copy() / 2
        state = self._update_state()

        return state

    def step (self, action: int):
        # cliping action into feasible action
        action = self.discrete_continuous[action].copy()

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
