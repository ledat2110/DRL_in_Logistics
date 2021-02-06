import json

config = {}

config['num_period'] = 1000
config['unit_cost'] = 100
config['production_capacity'] = 200
config['init_inventory'] = [10, 3, 40, 1]
config['storage_capacity'] = [100, 200, 200, 100]
config['product_price'] = [0, 100, 30, 20]
config['storage_cost'] = [100, 10, 100, 2]
config['penalty_cost'] = [0, 10, 30, 2]
config['truck_cost'] = [0, 10, 32, 18]
config['truck_capacity'] = [1, 10, 32, 23]
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

