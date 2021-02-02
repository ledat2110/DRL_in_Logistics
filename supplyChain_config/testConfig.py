import json

with open('./envConfig.json', 'r') as f:
    config = json.load(f)

factory = config['factory']
warehouses = config['warehouses']


print(factory)
for w in warehouses:
    print(w)
