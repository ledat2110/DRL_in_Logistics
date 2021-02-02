import numpy as np

from typing import Dict

class Warehouse:
    def __init__(self, config: Dict):
        self.id = config['id']
        self.init_inventory = config['init_inventory']
        self.inventory = config['init_inventory']
        self.capacity = config['capacity']
        self.storage_cost = config['storage_cost']
        self.penalty_cost = config['penalty_cost']
        self.truck_cost = config['truck_cost']
        self.truck_capacity = config['truck_capacity']

    def recieve(self, products):
        recieved = products

        if (self.inventory + products > self.capacity):
            recieved = self.capacity - self.inventory
            self.inventory = self.capacity
        else:
            self.inventory += products

        fee = self.truck_cost * (recieved * 1.0 / self.truck_capacity)
        return fee

    def demand(self, products):
        self.inventory -= products

    def storage_fee(self):
        return self.storage_cost * max(self.inventory, 0)

    def penalty_fee(self):
        return self.penalty_fee * min(self.inventory, 0)

    def reset(self):
        self.inventory = self.init_inventory
