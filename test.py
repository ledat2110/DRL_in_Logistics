from numpy.core.fromnumeric import mean
import pygame
from lib import common, envs, model
import argparse
import numpy as np
import drl
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

import torch

pygame.init()


def create_agent (env):
    action_dim = env.action_space.shape[0]

    eps = env.storage_capacity.copy() / 10
    Q = env.storage_capacity.copy()

    agent = model.ThresholdAgent(tuple(eps), tuple(Q), action_dim)

    return agent

def plot_fig (y, x, x_labels='steps', y_labels='reward', title='train_reward', legends = ['mean', 'max', 'median'], stepx=5, stepy=10, m=False):
    plt.figure(figsize=(12, 3))
    plt.plot(x, y)
    if m == True:
        mean_val = []
        sum_val = []
        for i in range(1, len(x)+1):
            mean_val.append(np.mean(y[:i]))
            sum_val.append(np.sum(y[:i]))
        plt.plot(x, mean_val)
        plt.plot(x, sum_val)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.xticks(np.arange(min(x), max(x) +1, stepx))
    if m == False:
        plt.yticks(np.arange(int(np.min(y)), int(np.max(y)) +1, stepy))
    if legends is not None:
        plt.legend(legends, bbox_to_anchor=(0, 1), loc='lower left', ncol=len(legends))
    plt.tight_layout()
    plt.savefig(f"/home/ledat/Desktop/result_png/{title}.png")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, help="type of actor model")
    parser.add_argument("-m", "--model", required=True, help="save dir of the model")
    parser.add_argument("-rm", "--retailer_model", default=None, help='save dir of the retailer model')
    parser.add_argument("-n", "--n_episode", default=1, help="the number of episode to play")
    parser.add_argument("-tr", "--trend", default=False, action='store_true', help='use trend demand')
    parser.add_argument("-v", "--var", default=False, action='store_true', help='use variance demand')
    parser.add_argument("-p", "--plot", default=False, action='store_true', help='show the plot of policy')
    parser.add_argument("-br", "--break_sp", default=False, action='store_true', help="")
    parser.add_argument("-rd", "--random_demand", default=True, action='store_false', help="")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(args.trend, args.var)
    assert args.type in ['2_agent', 'vpg', 'matrix_a2c', 'matrix_vpg', 'threshold']
    if args.type == 'threshold':
        env = envs.supply_chain.SupplyChain(
        #     n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10]),
        # periodic_demand=False, 
            m_demand=args.trend, v_demand=args.var,
            periodic_demand=args.random_demand,
            break_sp=args.break_sp
        )
        agent = create_agent(env)
    if args.type == 'vpg':
        env = envs.supply_chain.SupplyChain(
            m_demand=args.trend, v_demand=args.var,
            periodic_demand=args.random_demand,
            break_sp=args.break_sp
        )
        net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        net.load_state_dict(torch.load(args.model))
    if args.type == '2_agent':
        env = envs.supply_chain.SupplyChainRetailer()
        retailer_net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        if args.retailer_model is not None:
        	retailer_net.load_state_dict(torch.load(args.retailer_model))
        retailer_agent = model.NormalAgent(retailer_net, device=device)
        env = envs.supply_chain.SupplyChainWareHouse(
            m_demand=args.trend, v_demand=args.var,
            retailer_agent=retailer_agent,
            periodic_demand=args.random_demand,
            break_sp=args.break_sp
        )
        net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        net.load_state_dict(torch.load(args.model))
    if args.type == 'matrix_vpg':
        env = envs.supply_chain.SupplyChain(
            m_demand=args.trend, v_demand=args.var,
            periodic_demand=args.random_demand,
            break_sp=args.break_sp
        )
        net = model.MatrixModel2(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        net.load_state_dict(torch.load(args.model))
    if args.type != 'threshold':
        n_parameters = sum([np.prod(p.size()) for p in net.parameters()])
        print(net)
        print(n_parameters)
    
    invs = []
    actions = []
    demands = []
    rewards = []
    
    productions_cost = []
    trans_cost = []
    store_cost = []
    back_cost = []
    revenue = []
    total_rewards = []

    for eps in range(int(args.n_episode)):
            total_reward = 0
            obs = env.reset()
            while True:
                if args.type == 'threshold':
                    action = agent.get_action(obs)
                else:
                    obs_v = torch.FloatTensor([obs]).to(device)
                    mu_v, _ = net(obs_v)
                    action = mu_v.squeeze(dim=0).cpu().data.numpy()
                # action_t = env.clipping_action(action)
                demands.append(env.demand)                
                obs, reward, done, (action_t, inv) = env.step(action)
                invs.append(inv)
                actions.append(action_t)
                total_reward += reward
                rewards.append(reward)
                if done:
                    total_rewards.append(total_reward)
                    costs = env.get_result()
                    revenue.append(costs[0])
                    productions_cost.append(costs[1])
                    trans_cost.append(costs[2])
                    store_cost.append(costs[3])
                    back_cost.append(costs[4])
                    break
            if eps % 100 == 0:
                print(np.mean(total_rewards[-100:]))
    

    x = np.arange(0, env.num_period, 1)
    if args.plot:
        # print(invs)
        # print((rewards, revenue, cost))
        plot_fig(demands, x, y_labels='Đơn hàng yêu cầu', title=f'{args.type}_demand_run_{args.trend}_{args.var}_{args.break_sp}_{args.random_demand}', legends=['Cửa hàng 1', 'Cửa hàng 2', 'Cửa hàng 3'], stepx=5, stepy=1)
        plot_fig(actions, x, y_labels='Sản xuất và vận chuyển', title=f'{args.type}_act_run_{args.trend}_{args.var}_{args.break_sp}_{args.random_demand}', legends=['Sản xuất','Vận chuyển 1', 'Vận chuyển 2', 'Vận chuyển 3'], stepx=5, stepy=5)
        plot_fig(invs, x, y_labels='Kho hàng', title=f'{args.type}_inv_run_{args.trend}_{args.var}_{args.break_sp}_{args.random_demand}', legends=['Nhà phân phối','Cửa hàng 1', 'Cửa hàng 2', 'Cửa hàng 3'], stepx=5, stepy=10)
        # plot_fig(list(zip(rewards, revenue, cost)), x, y_labels='Lợi nhuận', title=f'{args.type}_reward_run_{args.trend}_{args.var}', legends=['Lợi nhuận', 'Lợi nhuận trung bình', 'Tổng lợi nhuận'], stepx=5, stepy=100)
    print("avg rewrards", np.around(np.mean(total_rewards)/100, 3))
    print("avg revenue", np.around(np.mean(revenue)/100, 3))
    print("avg prod cost", np.around(np.mean(productions_cost)/100, 3))
    print("avg trans cost", np.around(np.mean(trans_cost)/100, 3))
    print("avg store cost", np.around(np.mean(store_cost)/100, 3))
    print("avg back cost", np.around(np.mean(back_cost)/100, 3))

