import json

config = {}

config['seed'] = 2
config['num_period'] = 48
config['num_stores'] = 3
config['unit_cost'] = 1
config['production_capacity'] = 8
config['demand_max'] = 8
config['init_inventory'] = [10, 0, 0, 0]
config['storage_capacity'] = [20, 5, 5, 5]
config['product_price'] = [0, 30, 30, 30]
config['storage_cost'] = [0.01, 0.1, 0.1, 0.1]
config['penalty_cost'] = [5, 2, 2, 2]
config['truck_cost'] = [0, 1, 2, 3]
config['truck_capacity'] = [1, 2, 2, 2]
config['distribution'] = 'binom'
#config['factory'] = {
#        'warehouse': 1,
#        'unit_cost': 100,
#        'production_capacity': 200
#        }
#config['warehouses'] = []
#
#for i in range(4):
#    warehouse = {
#            'id':i,
#            'init_inventory': init_inventory[i],
#            'capacity': storage_capacity[i],
#            'price': prices[i],
#            'storage_cost': storage_cost[i],
#            'penalty_cost':penalty_cost[i],
#            'truck_cost': truck_cost[i],
#            'truck_capacity': truck_capacity[i]
#            }
#    config['warehouses'].append(warehouse)


with open('envConfig.json', 'w') as f:
    json.dump(config, f)

