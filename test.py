import matplotlib.pyplot as plt
import numpy as np
from lib import envs

env = envs.supply_chain.SupplyChain()
env.reset()
demands = env.get_demand()
print(demands)
x = np.arange(1, len(demands)+1, 1)
print(x)
plt.figure(figsize=(8, 2))
plt.plot(x, demands)
plt.yticks(np.arange(0, 10, 2))
plt.xlabel('steps')
plt.ylabel('demand')
plt.tight_layout()
plt.savefig("f1.png")
plt.show()
