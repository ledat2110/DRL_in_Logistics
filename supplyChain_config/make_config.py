import json

config = {}
warhouse_num = 4
storage_cost = [100, 10, 100, 2]
storage_capacity = [100, 10, 200, 3]
penalty_cost = [0, 10, 30, 2]
truck_cost = [0, 10, 32, 18]
truck_capacity = [1, 10, 32, 23]

config['factory'] = {
        'warehouse': 1,
        'unit_cost': 100,
        'production_capacity': 200
        }
config['warehouses'] = []

for i in range(4):
    warehouse = {
            'id':i,
            'capacity': storage_capacity[i],
            'storage_cost': storage_cost[i],
            'penalty_cost':penalty_cost[i],
            'truck_cost': truck_cost[i],
            'truck_capacity': truck_capacity[i]
            }
    config['warehouses'].append(warehouse)


with open('envConfig.json', 'w') as f:
    json.dump(config, f)

