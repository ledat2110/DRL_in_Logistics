#!/usr/bin/env python3
import os
import drl
import time
import argparse
from tensorboardX import SummaryWriter
import numpy as np

from lib import model, envs

import torch
import torch.optim as optim
import torch.nn.functional as F


GAMMA = 0.95
REWARD_STEPS = 1
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.01

#CLIP_GRAD = 0.1
REPLAY_SIZE = 600000
REPLAY_INITIAL = 10000

TEST_EPISODES = 60000
TEST_ITERS = 1000


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = drl.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-s", "--stop", action='store_false', default=True, help="stop when reach maximum episode")
    parser.add_argument("-sd", action='store_true', default=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    #envs = [drl.env.supply_chain.SupplyChain() for _ in range(ENV_COUNT)]
    env = envs.supply_chain.SupplyChain(
        # n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10]),
        m_demand=False, v_demand=False,
        # matrix_state=True
        )
    env2 = envs.supply_chain.SupplyChain(
        # n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10]),
        m_demand=True, v_demand=False,
        # matrix_state=True
        )
    env3 = envs.supply_chain.SupplyChain(
        # n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10]),
        m_demand=False, v_demand=True,
        # matrix_state=True
        )    
    test_env = envs.supply_chain.SupplyChain(
        # m_demand=True, v_demand=True,
        # n_stores=1, store_cost=np.array([0, 2]), truck_cost=np.array([3]),
        # storage_capacity=np.array([50, 10]), 
        m_demand=True, v_demand=True,

        # matrix_state=True
        )
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

    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    n_parameters = sum([np.prod(p.size()) for p in act_net.parameters()]) + sum([np.prod(p.size()) for p in crt_net.parameters()])
    print(n_parameters)
    tgt_act_net = drl.agent.TargetNet(act_net)
    tgt_crt_net = drl.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = model.AgentDDPG(act_net, env.storage_capacity, device=device)
    exp_source = drl.experience.ExperienceSourceFirstLast([env, env2, env3], agent, gamma=GAMMA, steps_count=1)
    buffer = drl.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    done_episodes = 0
    with drl.tracker.RewardTracker(writer) as tracker:
        with drl.tracker.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    # print(rewards[0])
                    tracker.reward(rewards[0], frame_idx)
                    done_episodes += 1

                if len(buffer) < REPLAY_INITIAL:
                    continue

                if done_episodes > TEST_EPISODES and args.stop:
                    break

                batch = buffer.sample(BATCH_SIZE)
                # print(batch)
                states_v, actions_v, rewards_v, dones_mask_v, last_states_v = drl.experience.unpack_batch_dqn(batch, device)
                # print(states_v)
                # if (torch.isnan(states_v)).sum() == 1 or (torch.isinf(states_v)).sum() == 1:
                #     print(states_v)
                #     quit()
                # if (torch.isnan(actions_v)).sum() == 1 or (torch.isinf(actions_v)).sum() == 1:
                #     print(actions_v)
                #     quit()
                # if (torch.isnan(rewards_v)).sum() == 1 or (torch.isinf(rewards_v)).sum() == 1:
                #     print(rewards_v)
                #     quit()
                # if (torch.isnan(last_states_v)).sum() == 1 or (torch.isinf(last_states_v)).sum() == 1:
                #     print(last_states_v)
                #     quit()


                # train critic
                crt_opt.zero_grad()

                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask_v] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA

                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()

                crt_opt.step()
                # tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                # tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()

                cur_actions_v = act_net(states_v)
                actor_loss_v = - crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()

                act_opt.step()
                # tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(1 - 1e-3)
                tgt_crt_net.alpha_sync(1 - 1e-3)


                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time() - ts, reward, step))
                    writer.add_scalar("test_reward", reward, frame_idx)
                    writer.add_scalar("test_step", reward, frame_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                            name = "best_%+.3f_%d.dat"%(reward, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = reward
    pass
