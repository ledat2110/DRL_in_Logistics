import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import drl
import gym
import time
import math
import argparse
import ptan
import os

from tensorboardX import SummaryWriter
from lib import model, envs, common

GAMMA = 0.95
LEARNING_RATE = 1e-4
ENTROPY_WEIGHT = 1e-4
BATCH_SIZE = 256
BASELINE_STEPS = 1000

REWARD_STEPS = 1
TEST_EPISODES = 1000
TRAIN_AGENT_EPISODES = 100
TRAIN_AGENT = 100
TEST_ITERS = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-sd", action='store_true', default=False)
    parser.add_argument("-ns", "--noisy", action='store_true', default=False, help="use the noisy linear layer")
    parser.add_argument("-c", "--count", action='store_true', default=False, help="use the psuedo counter")
    parser.add_argument("-s", "--stop", action='store_false', default=True)
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join("saves", "vpg1-"+args.name)
    os.makedirs(save_path, exist_ok=True)

    demand_net = model.A2CModel(4, 3).to(device)
    demand_agent = model.NormalAgent(demand_net, device)

    act_net = model.A2CModel(10, 4).to(device)
    act_agent = model.NormalAgent(act_net, device=device)
    

    env = envs.supply_chain.ManageSupplyChain(
        m_demand=False, v_demand=False, demand_agent=demand_agent
        )
    env2 = envs.supply_chain.ManageSupplyChain(
        m_demand=True, v_demand=False, demand_agent=demand_agent
        )
    env3 = envs.supply_chain.ManageSupplyChain(
        m_demand=False, v_demand=True, demand_agent=demand_agent
        )    
    test_env = envs.supply_chain.ManageSupplyChain(
        m_demand=True, v_demand=True, demand_agent=demand_agent
        # n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10])
        
    )

    d_env = envs.supply_chain.DemandSupplyChain(
        m_demand=False, v_demand=False, action_agent=act_agent
        )
    d_env2 = envs.supply_chain.DemandSupplyChain(
        m_demand=True, v_demand=False, action_agent=act_agent
        )
    d_env3 = envs.supply_chain.DemandSupplyChain(
        m_demand=False, v_demand=True, action_agent=act_agent
        )    
    d_test_env = envs.supply_chain.DemandSupplyChain(
        m_demand=True, v_demand=True, action_agent=act_agent
        # n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10])
        
    )

    if args.sd == True:
        print("supply distribution 10")
        env = envs.supply_distribution10.SupplyDistribution(
                n_stores=3, cap_truck=2, prod_cost=1, max_prod=3,
                store_cost=np.array([0, 2, 0, 0]),
                truck_cost=np.array([3, 3, 0]),
                cap_store=np.array([50, 10, 10, 10]),
                penalty_cost=1, price=2.5, gamma=1, max_demand=3, episode_length=25)
        test_env = envs.supply_distribution10.SupplyDistribution(
                n_stores=3, cap_truck=2, prod_cost=1, max_prod=3,
                store_cost=np.array([0, 2, 0, 0]),
                truck_cost=np.array([3, 3, 0]),
                cap_store=np.array([50, 10, 10, 10]),
                penalty_cost=1, price=2.5, gamma=1, max_demand=3, episode_length=25)

    if args.count:
        env = common.PseudoCountRewardWrapper(env, common.counts_hash, reward_scale=0.1)

    #net = model.LogisticsPGN(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # if args.noisy:
    #     act_net = model.NoisyActorModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    #     agent = model.NoisyA2CAgent(act_net, device)
    # else:
    #     act_net = model.ActorModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    #     agent = model.A2CAgent(act_net, device)
    # act_net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # if args.model is not None:
    #     act_net.load_state_dict(torch.load(args.model))
    #     print("load completed")
    # agent = model.A2CAgent(act_net, env, device)
    # print(act_net)
    # print(demand_net)
    # n_parameters = sum([np.prod(p.size()) for p in act_net.parameters()])
    # print(n_parameters)

    writer = SummaryWriter(comment=f'-cont_vpg-{args.name}')
    agent = model.A2CAgent(act_net, device=device)
    d_agent = model.A2CAgent(demand_net, device=device)

    exp_source = drl.experience.ExperienceSourceFirstLast(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)
    exp_source1 = drl.experience.ExperienceSourceFirstLast(d_env, d_agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    optimizer1 = optim.Adam(demand_net.parameters(), lr=LEARNING_RATE)

    done_episodes = 0

    baseline_buf = drl.common.utils.MeanBuffer(BASELINE_STEPS)
    baseline_buf1 = drl.common.utils.MeanBuffer(BASELINE_STEPS)
    batch = []
    batch1 = []
    best_reward = None
    total_train_rewards = []
    train_steps = []
    total_test_rewards = []
    test_steps = []
    t = 0
    with drl.tracker.RewardTracker(writer) as tracker:
        with drl.tracker.TBMeanTracker(writer, 10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source1):
                baseline_buf.add(exp.reward)
                baseline = baseline_buf.mean()

                batch.append(exp)
                
                rewards_steps = exp_source1.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tracker.reward(rewards[0], step_idx)
                    total_train_rewards.append(rewards[0]/1000)
                    # print(total_rewards)
                    train_steps.append(step_idx / 100000)
                    done_episodes += 1

                if done_episodes > TEST_EPISODES and args.stop:
                    break

                if len(batch) < BATCH_SIZE:
                    continue    

                states_v, actions_v, rewards_v, dones_v, last_states_v = drl.experience.unpack_batch_dqn(batch, device)
                batch.clear()

                scales_v = rewards_v - baseline

                optimizer1.zero_grad()
                mu_v, _ = demand_net(states_v)

                # log_prob_v = drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, actions_v)
                log_prob_v = common.cal_log_prob(mu_v, demand_net.logstd, actions_v)
                log_prob_v = scales_v.unsqueeze(-1) * log_prob_v
                loss_policy_v = -log_prob_v.mean()

                # if args.noisy:
                #     entropy_loss_v = 0
                # else:
                entropy_v = (-(torch.log(2 * math.pi * torch.exp(act_net.logstd)))).mean()
                entropy_loss_v = ENTROPY_WEIGHT * entropy_v
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                optimizer1.step()

                if done_episodes % TEST_ITERS == 0:
                    done_episodes1 = 0
                    rewards_buff = []
                    for step_idx1, exp1 in enumerate(exp_source):
                        baseline_buf1.add(exp.reward)
                        baseline1 = baseline_buf1.mean()

                        batch1.append(exp1)
                        
                        rewards_steps1 = exp_source.pop_rewards_steps()
                        if rewards_steps1:
                            rewards, steps = zip(*rewards_steps1)
                            rewards_buff.append(rewards[0])
                            if (done_episodes1 % 100 == 0):
                                print("done_episodes: %d, action_agent_reward: %.2f"%(done_episodes1, np.mean(rewards_buff[-100:])))
                            done_episodes1 += 1

                        if done_episodes1 > 1000:
                            break

                        if len(batch1) < BATCH_SIZE:
                            continue

                        states_v, actions_v, rewards_v, dones_v, last_states_v = drl.experience.unpack_batch_dqn(batch1, device)
                        batch1.clear()

                        scales_v = rewards_v - baseline1

                        optimizer.zero_grad()
                        mu_v, _ = act_net(states_v)

                        # log_prob_v = drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, actions_v)
                        log_prob_v = common.cal_log_prob(mu_v, act_net.logstd, actions_v)
                        log_prob_v = scales_v.unsqueeze(-1) * log_prob_v
                        loss_policy_v = -log_prob_v.mean()

                        # if args.noisy:
                        #     entropy_loss_v = 0
                        # else:
                        entropy_v = (-(torch.log(2 * math.pi * torch.exp(act_net.logstd)))).mean()
                        entropy_loss_v = ENTROPY_WEIGHT * entropy_v
                        loss_v = loss_policy_v + entropy_loss_v
                        loss_v.backward()
                        optimizer.step()