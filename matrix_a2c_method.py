import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils

import numpy as np
import drl
import gym
import time
import math
import argparse
import os

from tensorboardX import SummaryWriter
from lib import model, envs, common

GAMMA = 0.99
REWARD_STEPS = 1
BATCH_SIZE = 16
LEARNING_RATE_ACTOR =  1e-4
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-4

TEST_EPISODES = 10000
TEST_ITERS = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-sd", action='store_true', default=False)
    parser.add_argument("-s", "--stop", action='store_false', default=True, help="stop when reach mximum episode")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join("saves", "matrix_a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    #envs = [drl.env.supply_chain.SupplyChain() for _ in range(ENV_COUNT)]
    env = envs.supply_chain.SupplyChain(matrix_state=True)
    test_env = envs.supply_chain.SupplyChain(matrix_state=True)
    if args.sd == True:
        print("supply distribution 10")
        env = envs.supply_distribution10.SupplyDistribution(
                n_stores=3, cap_truck=2, prod_cost=1, max_prod=3,
                store_cost=np.array([0, 2, 0, 0]),
                truck_cost=np.array([3, 3, 0]),
                cap_store=np.array([50, 10, 10, 10]),
                penalty_cost=1,
                price=2.5,
                gamma=1, max_demand=3, episode_length=25
                )
        test_env = envs.supply_distribution10.SupplyDistribution(
                n_stores=3, cap_truck=2, prod_cost=1, max_prod=3,
                store_cost=np.array([0, 2, 0, 0]),
                truck_cost=np.array([3, 3, 0]),
                cap_store=np.array([50, 10, 10, 10]),
                penalty_cost=1,
                price=2.5,
                gamma=1, max_demand=3, episode_length=25
                )

    #net = model.LogisticsPGN(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    #agent = model.ContAgent(net, device)
    act_net = model.MatrixModel(env.observation_space.shape, env.action_space.shape[0]).to(device)
    print(act_net)
    n_parameters = sum([np.prod(p.size()) for p in act_net.parameters()])
    print(n_parameters)
    #crt_net = model.MatrixCriticModel(env.observation_space.shape).to(device)
    agent = model.A2CAgent(act_net, env, device)

    writer = SummaryWriter(comment=f'-matrix-a2c_{args.name}')
    exp_source = drl.experience.ExperienceSourceFirstLast(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    #optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    act_opitmizer = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_ACTOR)
    #crt_optimizer = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    done_episodes = 0
    best_rewards = None

    with drl.tracker.RewardTracker(writer, 1) as tracker:
        with drl.tracker.TBMeanTracker(writer, 10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tracker.reward(rewards[0], step_idx)
                    done_episodes += 1

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = common.test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time()-ts, reward, step))
                    writer.add_scalar("test_reward", reward, step_idx)
                    writer.add_scalar("test_step", step, step_idx)

                    if best_rewards is None or best_rewards < reward:
                        if best_rewards is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_rewards, reward))
                        name = "best_%+.3f_%d.dat"%(reward, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(act_net.state_dict(), fname)
                        best_rewards = reward

                if done_episodes > TEST_EPISODES and args.stop:
                    break

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = drl.experience.unpack_batch_a2c(batch, lambda x: act_net(x)[1], GAMMA ** REWARD_STEPS, device)
                batch.clear()

                #crt_optimizer.zero_grad()
                #value_v = crt_net(states_v)
                #loss_val_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                #loss_val_v.backward()
                #crt_optimizer.step()

                act_opitmizer.zero_grad()
                mu_v, val_v = act_net(states_v)
                loss_val_v = F.mse_loss(val_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - val_v.detach()
                # log_prob_v = adv_v * drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, actions_v)
                log_prob_v = adv_v * common.cal_log_prob(mu_v, act_net.logstd, actions_v)

                loss_policy_v = -log_prob_v.mean()
                loss_entropy_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * torch.exp(act_net.logstd)))).mean()

                loss_v = loss_policy_v + loss_entropy_v + loss_val_v
                loss_v.backward()
                act_opitmizer.step()

                #optimizer.zero_grad()
                #mu_v, var_v, val_v = net(states_v)
                #loss_value_v = F.mse_loss(val_v.squeeze(-1), vals_ref_v)

                #log_prob_v = calc_logprob(mu_v, var_v, actions_v)
                #adv_v = vals_ref_v.unsqueeze(-1) - val_v.detach()
                #log_prob_v = adv_v * log_prob_v
                #loss_policy_v = -log_prob_v.mean()

                #entropy_v = (-(torch.log(2 * math.pi * var_v + 1e-3) + 1) / 2).mean()
                #loss_entropy_v = ENTROPY_BETA * entropy_v

                #loss_v = loss_value_v + loss_entropy_v + loss_policy_v
                #loss_v.backward()
                ##nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                #optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", val_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("log_prob", log_prob_v, step_idx)
                tb_tracker.track("loss_entropy", loss_entropy_v, step_idx)
                tb_tracker.track("loss_value", loss_val_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss", loss_v, step_idx)

