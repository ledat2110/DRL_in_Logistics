import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.stats import *
import numpy as np
import math
import pygame
import collections
import torch

from PIL import Image
from typing import Tuple, List, Dict

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 40
SPEED = 1

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
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=2, max_production: int=15,
                    store_cost: np.array=np.array([2, 3, 3, 3], dtype=np.float32),
                    truck_cost: np.array=np.array([2, 2, 2], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 15, 20], dtype=np.float32),
                    penalty_cost: np.array=np.array([0, 2, 2, 2], dtype=np.float32), price: int=20,
                    max_demand: int=4, num_period: int=52, periodic_demand: bool=True,
                    matrix_state: bool=False, v_demand: bool=False, m_demand: bool=False,
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
        self.var_demand = v_demand
        self.trend_demand = m_demand
        self.v_demand = self.m_demand = 0
        self.total_pro_cost = 0
        self.total_trans_cost = 0
        self.total_store_cost = 0
        self.total_back_cost = 0
        self.total_revenue = 0
        

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
        self.inventory = self.storage_capacity.copy() / 2
        self.old_demand = self.demand.copy()
        state = self._update_state()
        self._update_demand()
        self.sum_reward = 0
        self.v_demand = np.random.uniform(1/3, 3) if self.var_demand == True else 0
        self.m_demand = np.random.uniform(-self.demand_max/2, self.demand_max/2) if self.trend_demand == True else 0
        self.total_pro_cost = 0
        self.total_trans_cost = 0
        self.total_store_cost = 0
        self.total_back_cost = 0
        self.total_revenue = 0

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

        return state, reward, done, (action, self.inventory)

    def clipping_action (self, action: np.ndarray) -> np.ndarray:
        upper_bound = self.storage_capacity - self.inventory
        action = np.clip(action, np.zeros(self.action_dim), upper_bound)
        action[0] = min(self.production_capacity, action[0])
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

        revenue = np.sum(np.minimum(np.maximum(self.inventory[1:], np.zeros_like(self.demand)), self.demand)  * self.product_price)
        production_cost = self.unit_cost * action[0]
        storage_cost = np.sum(np.maximum(zeros_array, self.inventory) * self.storage_cost)
        penalty_cost = np.sum(np.minimum(zeros_array, self.inventory) * self.penalty_cost)
        truck_cost = np.sum(np.ceil(action[1:]/self.truck_capacity) * self.truck_cost)

        reward = revenue - production_cost - storage_cost + penalty_cost - truck_cost
        
        self.total_store_cost += storage_cost
        self.total_back_cost -= penalty_cost
        self.total_pro_cost += production_cost
        self.total_trans_cost += truck_cost
        self.total_revenue += revenue
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
        # self.m_demand = 2
        # self.v_demand = 0
        for i in range(self.num_stores):
            eps = np.random.uniform(-1, 2)
            # eps = 0
            # eps = np.random.choice([-1, 0, 1], p=[1/3, 1/3, 1/3])
            # print(eps)
            rad = np.pi * (self.period + 2 * i) / (.25 * self.num_period) 
            v = self.demand_max * (self.v_demand ** (self.period / self.num_period)) \
                * np.sin(rad) / 4
            m = self.demand_max + self.m_demand * (self.period / self.num_period)
            demand[i] = np.floor(m + v + eps)
            # if self.periodic_demand == True:
            #     eps = np.random.choice([0, 1], p=(0.5, 0.5))
                # rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
            #     val = .5 * self.demand_max * np.sin(rad) * (self.v_demand ** (self.period / self.num_period)) + .5 * self.demand_max + self.m_demand * (self.period / self.num_period) + eps
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

    def get_result (self):
        return (self.total_revenue, self.total_pro_cost, self.total_trans_cost,\
            self.total_store_cost, self.total_back_cost)

class SupplyChainWareHouse (gym.Env):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=2, max_production: int=15,
                    store_cost: np.array=np.array([2, 3, 3, 3], dtype=np.float32),
                    truck_cost: np.array=np.array([2, 2, 2], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 15, 20], dtype=np.float32),
                    penalty_cost: np.array=np.array([0, 2, 2, 2], dtype=np.float32), price: int=20,
                    max_demand: int=4, num_period: int=52, periodic_demand: bool=True,
                    matrix_state: bool=False, v_demand: bool=False, m_demand: bool=False, retailer_agent=None,
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
        self.var_demand = v_demand
        self.trend_demand = m_demand
        self.v_demand = self.m_demand = 0
        self.total_pro_cost = 0
        self.total_trans_cost = 0
        self.total_store_cost = 0
        self.total_back_cost = 0
        self.total_revenue = 0
        self.retailer_agent = retailer_agent
        

        self.action_dim = 1
        self.action_space = gym.spaces.Box(
                low = np.zeros(self.action_dim),
                high = self.storage_capacity[0],
                dtype = np.float32
                )

        self.demand_dim = self.num_stores
        self.matrix_state = matrix_state
        if self.matrix_state == False:
            self.state_dim = 1 + self.demand_dim * 2
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
        self.retailer_demand = np.zeros(self.num_stores, dtype=int)
        self.old_retailer_demand = self.retailer_demand.copy()
        self.inventory = self.storage_capacity.copy() / 2
        self.old_demand = self.demand.copy()
        state = self._update_state()
        self._update_demand()
        self.sum_reward = 0
        self.v_demand = np.random.uniform(1/3, 3) if self.var_demand == True else 0
        self.m_demand = np.random.uniform(-self.demand_max/2, self.demand_max/2) if self.trend_demand == True else 0
        self.total_pro_cost = 0
        self.total_trans_cost = 0
        self.total_store_cost = 0
        self.total_back_cost = 0
        self.total_revenue = 0

        return state

    def step (self, w_action: np.ndarray):
        # cliping action into feasible action
        self.old_retailer_demand = self.retailer_demand.copy()
        self.retailer_demand, _ = self.retailer_agent(self.retailer_state, self.retailer_state)
       
        action = np.concatenate((w_action, self.retailer_demand))
        action = self.clipping_action(action)
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

        return state, reward, done, (action, self.inventory)

    def clipping_action (self, action: np.ndarray) -> np.ndarray:
        upper_bound = self.storage_capacity - self.inventory
        action = np.clip(action, np.zeros(self.action_dim), upper_bound)
        action[0] = min(self.production_capacity, action[0])
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

        revenue = np.sum(np.minimum(np.maximum(self.inventory[1:], np.zeros_like(self.demand)), self.demand)  * self.product_price)
        production_cost = self.unit_cost * action[0]
        storage_cost = np.sum(np.maximum(zeros_array, self.inventory) * self.storage_cost)
        penalty_cost = np.sum(np.minimum(zeros_array, self.inventory) * self.penalty_cost)
        truck_cost = np.sum(np.ceil(action[1:]/self.truck_capacity) * self.truck_cost)

        reward = revenue - production_cost - storage_cost + penalty_cost - truck_cost
        
        self.total_store_cost += storage_cost
        self.total_back_cost -= penalty_cost
        self.total_pro_cost += production_cost
        self.total_trans_cost += truck_cost
        self.total_revenue += revenue
        return reward

    def _update_state (self):
        inventory = self.inventory.copy()
        retailer_demand = self.retailer_demand.copy()
        old_retailer_demand = self.old_retailer_demand.copy()
        if self.matrix_state == False:
            inv = np.array([inventory[0]])
            state = np.concatenate((inv, retailer_demand, old_retailer_demand)).astype(np.float32)
        
        demand = self.demand.copy()
        old_demand = self.old_demand.copy()
        self.retailer_state  = np.concatenate((inventory[1:], demand, old_demand)).astype(np.float32)

        # print(state)

        return state

    def _update_demand (self):
        demand = np.zeros(self.num_stores, dtype=int)
        # self.m_demand = 2
        # self.v_demand = 0
        for i in range(self.num_stores):
            eps = np.random.uniform(-1, 2)
            # eps = 0
            # eps = np.random.choice([-1, 0, 1], p=[1/3, 1/3, 1/3])
            # print(eps)
            rad = np.pi * (self.period + 2 * i) / (.25 * self.num_period) 
            v = self.demand_max * (self.v_demand ** (self.period / self.num_period)) \
                * np.sin(rad) / 4
            m = self.demand_max + self.m_demand * (self.period / self.num_period)
            demand[i] = np.floor(m + v + eps)
            # if self.periodic_demand == True:
            #     eps = np.random.choice([0, 1], p=(0.5, 0.5))
                # rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
            #     val = .5 * self.demand_max * np.sin(rad) * (self.v_demand ** (self.period / self.num_period)) + .5 * self.demand_max + self.m_demand * (self.period / self.num_period) + eps
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

    def get_result (self):
        return (self.total_revenue, self.total_pro_cost, self.total_trans_cost,\
            self.total_store_cost, self.total_back_cost)

class SupplyChainRetailer (gym.Env):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, 
                    store_cost: np.array=np.array([0, 3, 3, 3], dtype=np.float32),
                    truck_cost: np.array=np.array([2, 2, 2], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 15, 20], dtype=np.float32),
                    penalty_cost: np.array=np.array([0, 2, 2, 2], dtype=np.float32), price: int=20,
                    max_demand: int=4, num_period: int=52, periodic_demand: bool=True,
                    matrix_state: bool=False, v_demand: bool=False, m_demand: bool=False,
                    disp: bool=False):
        self.num_period = num_period
        # self.unit_cost = production_cost
        # self.production_capacity = max_production
        self.storage_capacity = storage_capacity
        self.product_price = price
        self.storage_cost = store_cost
        self.penalty_cost = penalty_cost
        self.truck_cost = truck_cost
        self.truck_capacity = cap_truck
        self.demand_max = max_demand
        self.num_stores = n_stores
        self.periodic_demand = periodic_demand
        self.var_demand = v_demand
        self.trend_demand = m_demand
        self.v_demand = self.m_demand = 0
        self.total_pro_cost = 0
        self.total_trans_cost = 0
        self.total_store_cost = 0
        self.total_back_cost = 0
        self.total_revenue = 0
        

        self.demand_dim = self.num_stores
        self.action_dim = self.demand_dim
        self.action_space = gym.spaces.Box(
                low = np.zeros(self.action_dim),
                high = self.storage_capacity[1:],
                dtype = np.float32
                )

        self.matrix_state = matrix_state
        if self.matrix_state == False:
            self.state_dim = self.demand_dim * 3
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
        self.inventory = self.storage_capacity.copy() / 2
        self.old_demand = self.demand.copy()
        self._update_demand()
        state = self._update_state()
        self.sum_reward = 0
        self.v_demand = np.random.uniform(1/3, 3) if self.var_demand == True else 0
        self.m_demand = np.random.uniform(-self.demand_max/2, self.demand_max/2) if self.trend_demand == True else 0
        self.total_pro_cost = 0
        self.total_trans_cost = 0
        self.total_store_cost = 0
        self.total_back_cost = 0
        self.total_revenue = 0

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
        upper_bound = self.storage_capacity[1:] - self.inventory[1:]
        action = np.clip(action, np.zeros(self.action_dim), upper_bound)
        # action[0] = min(self.production_capacity, action[0])
        if np.sum(action) > self.inventory[0] and np.sum(action) != 0:
            action = action * self.inventory[0] / np.sum(action)
        action = np.around(action, 4).astype(np.float32)

        return action

    def _update_inventory (self, action: np.ndarray):
        self.inventory[0] = min(self.inventory[0] + 5 - np.sum(action[1:]), self.storage_capacity[0])
        self.inventory[0] = max(25, self.inventory[0])
        self.inventory[1:] = np.minimum(self.inventory[1:] + action- self.demand, self.storage_capacity[1:])
        self.inventory = np.around(self.inventory, 4).astype(np.float32)

    def _update_reward (self, action: np.ndarray) -> float:
        zeros_array = np.zeros_like(self.inventory)

        revenue = np.sum(np.minimum(np.maximum(self.inventory[1:], np.zeros_like(self.demand)), self.demand)  * self.product_price)
        # production_cost = self.unit_cost * action[0]
        storage_cost = np.sum(np.maximum(zeros_array, self.inventory) * self.storage_cost)
        penalty_cost = np.sum(np.minimum(zeros_array, self.inventory) * self.penalty_cost)
        truck_cost = np.sum(np.ceil(action/self.truck_capacity) * self.truck_cost)

        reward = revenue - storage_cost + penalty_cost - truck_cost
        
        self.total_store_cost += storage_cost
        self.total_back_cost -= penalty_cost
        # self.total_pro_cost += production_cost
        self.total_trans_cost += truck_cost
        self.total_revenue += revenue
        return reward

    def _update_state (self):
        inventory = self.inventory.copy()
        demand = self.demand.copy()
        old_demand = self.old_demand.copy()
        if self.matrix_state == False:
            state = np.concatenate((inventory[1:], demand, old_demand)).astype(np.float32)
        else:
            state = []
            state.append(inventory[1:])
            demand = np.concatenate([[0], demand]).astype(np.float32)
            old_demand = np.concatenate([[0], old_demand]).astype(np.float32)
            state.append(demand)
            state.append(old_demand)
            state = np.array(state, dtype=np.float32, copy=False)

        return state

    def _update_demand (self):
        demand = np.zeros(self.num_stores, dtype=int)
        # self.m_demand = 2
        # self.v_demand = 0
        for i in range(self.num_stores):
            eps = np.random.uniform(-1, 2)
            # eps = 0
            # eps = np.random.choice([-1, 0, 1], p=[1/3, 1/3, 1/3])
            # print(eps)
            rad = np.pi * (self.period + 2 * i) / (.25 * self.num_period) 
            v = self.demand_max * (self.v_demand ** (self.period / self.num_period)) \
                * np.sin(rad) / 4
            m = self.demand_max + self.m_demand * (self.period / self.num_period)
            demand[i] = np.floor(m + v + eps)
            # if self.periodic_demand == True:
            #     eps = np.random.choice([0, 1], p=(0.5, 0.5))
                # rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
            #     val = .5 * self.demand_max * np.sin(rad) * (self.v_demand ** (self.period / self.num_period)) + .5 * self.demand_max + self.m_demand * (self.period / self.num_period) + eps
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

    def get_result (self):
        return (self.total_revenue, self.total_back_cost, self.total_pro_cost,\
            self.total_store_cost, self.total_trans_cost)

class ManageSupplyChain (SupplyChain):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=2, max_production: int=15,
                    store_cost: np.array=np.array([2, 1, 2, 1], dtype=np.float32),
                    truck_cost: np.array=np.array([2, 2, 2], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 15, 20], dtype=np.float32),
                    penalty_cost: np.array=np.array([0, 4, 3, 4], dtype=np.float32), price: int=20,
                    max_demand: int=4, num_period: int=52, periodic_demand: bool=True,
                    matrix_state: bool=False, v_demand: bool=False, m_demand: bool=False,
                    disp: bool=False, demand_agent=None):
        self.demand_agent = demand_agent
        super().__init__(n_stores=n_stores, cap_truck=cap_truck, production_cost=production_cost, max_production=max_production, store_cost=store_cost, truck_cost=truck_cost, storage_capacity=storage_capacity, penalty_cost=penalty_cost, price=price, max_demand=max_demand, num_period=num_period, periodic_demand=periodic_demand, matrix_state=matrix_state, v_demand=v_demand, m_demand=m_demand, disp=disp)

    def _update_demand (self):
        demand_state = self.inventory.copy()
        demand, _ = self.demand_agent(demand_state, demand_state)
        demand = np.clip(demand, np.zeros_like(demand), self.demand_max)
        # demand = np.zeros(self.num_stores, dtype=int)
        # # self.m_demand = 2
        # # self.v_demand = 0
        # for i in range(self.num_stores):
        #     eps = np.random.uniform(-1, 2)
        #     # eps = 0
        #     # eps = np.random.choice([-1, 0, 1], p=[1/3, 1/3, 1/3])
        #     # print(eps)
        #     rad = np.pi * (self.period + 2 * i) / (.25 * self.num_period) 
        #     v = self.demand_max * (self.v_demand ** (self.period / self.num_period)) \
        #         * np.sin(rad) / 4
        #     m = self.demand_max + self.m_demand * (self.period / self.num_period)
        #     demand[i] = np.floor(m + v + eps)
        #     # if self.periodic_demand == True:
        #     #     eps = np.random.choice([0, 1], p=(0.5, 0.5))
        #         # rad = np.pi * (self.period + 2 * i) / (.5 * self.num_period) - np.pi
        #     #     val = .5 * self.demand_max * np.sin(rad) * (self.v_demand ** (self.period / self.num_period)) + .5 * self.demand_max + self.m_demand * (self.period / self.num_period) + eps
        #     #     demand[i] = int(np.floor(val))
        #     # else:
        #     #     demand[i] = np.random.randint(low=0, high=self.demand_max)

        self.demand = demand 

class DemandSupplyChain (SupplyChain):
    def __init__ (self, n_stores: int=3, cap_truck: int=2, production_cost: int=2, max_production: int=15,
                    store_cost: np.array=np.array([2, 1, 2, 1], dtype=np.float32),
                    truck_cost: np.array=np.array([2, 2, 2], dtype=np.float32),
                    storage_capacity: np.array=np.array([50, 10, 15, 20], dtype=np.float32),
                    penalty_cost: np.array=np.array([0, 4, 3, 4], dtype=np.float32), price: int=20,
                    max_demand: int=4, num_period: int=52, periodic_demand: bool=True,
                    matrix_state: bool=False, v_demand: bool=False, m_demand: bool=False,
                    disp: bool=False, action_agent=None):
        self.action_agent = action_agent
        super().__init__(n_stores=n_stores, cap_truck=cap_truck, production_cost=production_cost, max_production=max_production, store_cost=store_cost, truck_cost=truck_cost, storage_capacity=storage_capacity, penalty_cost=penalty_cost, price=price, max_demand=max_demand, num_period=num_period, periodic_demand=periodic_demand, matrix_state=matrix_state, v_demand=v_demand, m_demand=m_demand, disp=disp)

    def reset(self):
        super().reset()
        return self.inventory.copy()

    def step (self, demand: np.ndarray):
        self.old_demand = self.demand

        # update state
        state = self._update_state()

        action, _ = self.action_agent(state, state)
        action = action[0]
        # cliping action into feasible action
        action = self.clipping_action(action)
        # print(action)
        demand = np.clip(demand, np.zeros_like(demand), self.demand_max)
        # print(self.period, demand)
        self.demand = demand

        # update inventory
        self._update_inventory(action)

        # update reward
        reward = -self._update_reward(action)
        info = "Demand was: ", self.demand
        self.sum_reward += reward

        

        # # update historical demand
        # self.old_demand = self.demand.copy()

        # render
        if self.disp:
            self.render(action)
            self.clock.tick(SPEED)

        # update t
        self.period += 1

        # # update demand
        # self.demand

        # set done
        done = False
        if self.period >= self.num_period:
            done = True

        state = self.inventory.copy()


        return state, reward, done, info

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
