import pygame
from lib import envs, model
import argparse
import numpy as np

import torch

pygame.init()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, help="type of actor model")
    parser.add_argument("-m", "--model", required=True, help="save dir of the model")
    args = parser.parse_args()

    assert args.type in ['a2c', 'vpg', 'matrix_a2c', 'matrix_vpg']
    if args.type == 'vpg':
        env = envs.supply_chain.SupplyChain(disp=True)
        net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0])
        net.load_state_dict(torch.load(args.model))
    if args.type == 'matrix_vpg':
        env = envs.supply_chain.SupplyChain(matrix_state=True, disp=True)
        net = model.MatrixModel(env.observation_space.shape, env.action_space.shape[0])
        net.load_state_dict(torch.load(args.model))
    obs = env.reset()
    while True:
        
        print(obs)
        obs_v = torch.FloatTensor([obs])
        mu_v, _ = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = env.clipping_action(action)
        print(action)
        obs, _, done, _ = env.step(action)
        if done:
            break
    while True:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
