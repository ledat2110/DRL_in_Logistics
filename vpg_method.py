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

GAMMA = 0.99
LEARNING_RATE = 1e-5
ENTROPY_WEIGHT = 1e-4
BATCH_SIZE = 16
BASELINE_STEPS = 1000

REWARD_STEPS = 2
TEST_EPISODES = 10000
TEST_ITERS = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-sd", action='store_true', default=False)
    parser.add_argument("-ns", "--noisy", action='store_true', default=False, help="use the noisy linear layer")
    parser.add_argument("-c", "--count", action='store_true', default=False, help="use the psuedo counter")
    parser.add_argument("-s", "--stop", action='store_false', default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join("saves", "vpg-"+args.name)
    os.makedirs(save_path, exist_ok=True)

    env = envs.supply_chain.SupplyChain()
    test_env = envs.supply_chain.SupplyChain()
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
    act_net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
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
                    done_episodes += 1
                    # reward = rewards[0]

                    # if best_reward is None or best_reward < reward:
                    #     if best_reward is not None:
                    #         print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                    #         name = "best_%+.3f_%d.dat"%(reward, step_idx)
                    #         fname = os.path.join(save_path, name)
                    #         torch.save(act_net.state_dict(), fname)
                    #     best_reward = reward

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = common.test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time()-ts, reward, step))
                    writer.add_scalar("test_reward", reward, step_idx)
                    writer.add_scalar("test_step", step, step_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                            name = "best_%+.3f_%d.dat"%(reward, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = reward

                if done_episodes > TEST_EPISODES and args.stop:
                    break

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, rewards_v, dones_v, last_states_v = drl.experience.unpack_batch_dqn(batch, device)
                batch.clear()

                scales_v = rewards_v - baseline

                optimizer.zero_grad()
                mu_v, _ = act_net(states_v)

                log_prob_v = drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, actions_v)
                # log_prob_v = common.cal_log_prob(mu_v, act_net.logstd, actions_v)
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

                tb_tracker.track("baseline", baseline, step_idx)
                tb_tracker.track("entropy", entropy_v, step_idx)
                tb_tracker.track("advantage", scales_v, step_idx)
                tb_tracker.track("log_prob", log_prob_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss", loss_v, step_idx)

