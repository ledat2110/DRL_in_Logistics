import json

with open ('../env_config/envConfig.json', 'r') as f:
    config = json.load(f)
for key in config.keys():
    print(key, config[key])
