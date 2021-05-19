import pygame
from lib import envs, model
import argparse
import numpy as np
import drl

from tensorboardX import SummaryWriter

import torch

pygame.init()


def create_agent (env):
    action_dim = env.action_space.shape[0]

    eps = env.storage_capacity.copy() / 10
    Q = env.storage_capacity.copy()

    agent = model.ThresholdAgent(tuple(eps), tuple(Q), action_dim)

    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, help="type of actor model")
    parser.add_argument("-m", "--model", required=True, help="save dir of the model")
    parser.add_argument("-n", "--n_episode", default=1, help="the number of episode to play")
    parser.add_argument("-d", "--display", action='store_true', default=False)
    args = parser.parse_args()

    assert args.type in ['a2c', 'vpg', 'matrix_a2c', 'matrix_vpg', 'threshold']
    if args.type == 'threshold':
        env = envs.supply_chain.SupplyChain(
        #     n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10]),
        periodic_demand=False,Dem
        disp=args.display)
        agent = create_agent(env)
    if args.type == 'vpg':
        env = envs.supply_chain.SupplyChain(disp=args.display, periodic_demand=False)
        net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0])
        net.load_state_dict(torch.load(args.model))
    if args.type == 'matrix_vpg':
        env = envs.supply_chain.SupplyChain(disp=args.display, matrix_state=True, periodic_demand=False)
        net = model.MatrixModel(env.observation_space.shape, env.action_space.shape[0])
        net.load_state_dict(torch.load(args.model))
    writer = SummaryWriter(logdir=f"reward_play/{args.type}_{args.n_episode}", comment=f"_{args.type}_{args.n_episode}")

    with drl.tracker.RewardTracker(writer) as tracker:
        for eps in range(int(args.n_episode)):
            total_reward = 0
            obs = env.reset()
            while True:
                if args.type == 'threshold':
                    action = agent.get_action(obs)
                else:
                    obs_v = torch.FloatTensor([obs])
                    mu_v, _ = net(obs_v)
                    action = mu_v.squeeze(dim=0).data.numpy()
                action = env.clipping_action(action)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    if args.display:
                        env.save_img(f'play_result_{args.type}.png')
                    break

            tracker.reward(total_reward, eps)
    while args.display:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
