import numpy as np
from typing import Dict, Tuple, List

class Factory:
    def __init__(self, config: Dict):
        self.products = 0
        self.production_capacity = config['production_capacity']
        self.unit_cost = config['unit_cost']

    def produce(self, products):
        produced = products

        if self.products + products < self.production_capacity:
            self.products += products
        else:
            produced = self.production_capacity - self.products
            self.products = self.production_capacity

        return self.unit_cost * produced

    def reset(self):
        self.products = 0
