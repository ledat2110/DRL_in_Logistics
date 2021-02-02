import numpy as np

class Warehouse:
    def __init__(self, inventory_max: int, storage_cost: float, penalty_cost: float, deliver_cost: float, deliver_volume: float):
        self.inventory_max = inventory_max
        self.inventory = 0
        self.penalty_cost = penalty_cost
        self.storage_cost = storage_cost
        self.deliver_cost = deliver_cost
        self.deliver_volume = deliver_volume


    def recieve(self, products):
        fee = 0
        if (self.inventory + products > self.inventory_max):
            fee = self.deliver_cost * (self.inventory_max - self.inventory) * 1.0 / self.deliver_volume
            self.inventory = self.inventory_max
        else:
            fee = self.deliver_cost * products * 1.0 / self.deliver_volume
            self.inventory += products

        return fee

    def demand(self, products):
        self.inventory -= products

    def storage_fee(self):
        return self.storage_cost * max(self.inventory, 0)

    def penalty_fee(self):
        return self.penalty_fee * min(self.inventory, 0)

