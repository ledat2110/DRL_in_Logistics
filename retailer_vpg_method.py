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
TEST_EPISODES = 60000
TEST_ITERS = 10000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-s", "--stop", action='store_false', default=True, help="stop when reach maximum episode")
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join("saves", "retailer_vpg-"+args.name)
    os.makedirs(save_path, exist_ok=True)
   
    env = envs.supply_chain.SupplyChainRetailer(
        m_demand=False, v_demand=False
        )
    env2 = envs.supply_chain.SupplyChainRetailer(
        m_demand=True, v_demand=False
        )
    env3 = envs.supply_chain.SupplyChainRetailer(
        m_demand=False, v_demand=True
        )    
    test_env = envs.supply_chain.SupplyChainRetailer(
        m_demand=True, v_demand=True
    )
    
    act_net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    if args.model is not None:
        act_net.load_state_dict(torch.load(args.model))
        print("load completed")
    agent = model.A2CAgent(act_net, env, device)
    print(act_net)
    n_parameters = sum([np.prod(p.size()) for p in act_net.parameters()])
    print(n_parameters)

    writer = SummaryWriter(comment=f'-cont_vpg-{args.name}')
    exp_source = drl.experience.ExperienceSourceFirstLast(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)

    done_episodes = 0

    baseline_buf = drl.common.utils.MeanBuffer(BASELINE_STEPS)
    batch = []
    best_reward = None
    total_train_rewards = []
    train_steps = []
    total_test_rewards = []
    test_steps = []
    t = 0
    with drl.tracker.RewardTracker(writer) as tracker:
        with drl.tracker.TBMeanTracker(writer, 10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                baseline_buf.add(exp.reward)
                baseline = baseline_buf.mean()

                batch.append(exp)

                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tracker.reward(rewards[0], step_idx)
                    total_train_rewards.append(rewards[0]/1000)
                    # print(total_rewards)
                    train_steps.append(step_idx / 100000)
                    done_episodes += 1

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = common.test_net(act_net, test_env, device=device)
                    
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time()-ts, reward, step))
                    writer.add_scalar("test_reward", reward, step_idx)
                    writer.add_scalar("test_step", step, step_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                        
                        best_reward = reward
                    name = "best_%+.3f_%d.dat"%(reward, step_idx)
                    fname = os.path.join(save_path, name)
                    torch.save(act_net.state_dict(), fname)
                    total_test_rewards.append(reward/1000)
                    test_steps.append(t)
                    t += 1
                    

                if done_episodes > TEST_EPISODES and args.stop:
                    common.plot_fig(total_train_rewards, train_steps,y_labels='reward x 1000', x_labels='steps x 100000', title="vpg_train_reward", stepx=2, stepy=2)
                    common.plot_fig(total_test_rewards, test_steps,y_labels='reward x 1000', x_labels='tests', title="vpg_test_reward", stepx=50, stepy=2)
                    break

                if len(batch) < BATCH_SIZE:
                    continue    

                states_v, actions_v, rewards_v, dones_v, last_states_v = drl.experience.unpack_batch_dqn(batch, device)
                batch.clear()

                scales_v = rewards_v - baseline

                optimizer.zero_grad()
                mu_v, _ = act_net(states_v)

                log_prob_v = common.cal_log_prob(mu_v, act_net.logstd, actions_v)
                log_prob_v = scales_v.unsqueeze(-1) * log_prob_v
                loss_policy_v = -log_prob_v.mean()

                entropy_v = (-(torch.log(2 * math.pi * torch.exp(act_net.logstd)))).mean()
                entropy_loss_v = ENTROPY_WEIGHT * entropy_v
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                optimizer.step()

