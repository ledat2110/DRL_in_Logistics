import numpy as np

from typing import Dict

class Warehouse:
    def __init__(self, config: Dict):
        self.id = config['id']
        self.init_inventory = config['init_inventory']
        self.inventory = config['init_inventory']
        self.capacity = config['capacity']
        self.storage_fee = config['storage_cost']
        self.penalty_fee = config['penalty_cost']
        self.truck_cost = config['truck_cost']
        self.truck_capacity = config['truck_capacity']
        self.price = config['price']

    def transportation_cost(self, products: int) -> float:
        recieved = products

        if (self.inventory + products > self.capacity):
            recieved = self.capacity - self.inventory
            self.inventory = self.capacity
        else:
            self.inventory += products

        cost = self.truck_cost * (recieved * 1.0 / self.truck_capacity)
        return cost

    def revenue(self, products: int) -> float:
        sold_products = products
        if (products > self.inventory):
            sold_products = self.inventory

        self.inventory -= products
        return self.price * self.products

    def storage_cost(self) -> float:
        return self.storage_fee * max(self.inventory, 0)

    def penalty_cost(self) -> float:
        return self.penalty_fee * min(self.inventory, 0)

    def reset(self):
        self.inventory = self.init_inventory
