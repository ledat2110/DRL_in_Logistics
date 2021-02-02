import numpy as np

class Factory:
    def __init__(self, p_max: int, produce_cost: float):
        self.p_max = p_max
        self.products = 0
        self.produce_cost = produce_cost

    def produce(self, products):
        fee = 0
        if self.products + products < self.p_max:
            self.products += products
            fee = self.produce_cost * products
        else:
            fee = self.produce_cost * (self.p_max - self.products)
            self.products = self.p_max

        return fee
