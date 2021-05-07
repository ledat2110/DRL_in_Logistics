import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import drl
import gym
import time
import math

from lib import envs
from tensorboardX import SummaryWriter

GAMMA = 1
LEARNING_RATE = 1e-4
ENTROPY_WEIGHT = 0.01
BATCH_SIZE = 16
BASELINE_STEPS = 1000

REWARD_STEPS = 1
GRAD_L2_CLIP = 0.1
TEST_EPISODES = 10000

ENV_COUNT = 4

class LogisticsPGN (nn.Module):
    def __init__ (self, ob_dim, action_dim):
        super(LogisticsPGN, self).__init__ ()

        self.base = nn.Sequential(
                nn.Linear(ob_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()

                )

        self.prob = nn.Sequential(
                nn.Linear(64, action_dim)
                )

    def _format (self, x):
        fx = x
        if not isinstance(fx, torch.Tensor):
            fx = torch.Tensor(x, dtype=torch.float32)
            fx = fx.unsqueeze(0)
        fx = fx.float()
        return fx

    def forward (self, x):
        fx = self._format(x)
        out_base = self.base(fx)
        out_prob = self.prob(out_base)

        return out_prob

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #envs = [drl.env.supply_chain.DiscreteSupplyChain() for _ in range(ENV_COUNT)]
    env = envs.supply_distribution3.SupplyDistribution(
            n_stores=3, cap_truck=2, prod_cost=1, max_prod=3,
            store_cost=np.array([0, 2, 0, 0]),
            truck_cost=np.array([3, 3, 0]),
            cap_store=np.array([50, 10, 10, 10]),
            penalty_cost=1, price=2.5, gamma=1, max_demand=3, episode_length=25
            )
    net = LogisticsPGN(env.observation_dim(), env.action_space.n).to(device)
    agent = drl.agent.PolicyAgent(net, device=device, apply_softmax=True)

    writer = SummaryWriter(comment='-vpg')
    exp_source = drl.experience.ExperienceSource(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    batch = drl.experience.BatchData(max_size=BATCH_SIZE)

    baseline_buf = drl.utils.MeanBuffer(BASELINE_STEPS)
    done_episodes = 0
    with drl.tracker.RewardTracker(writer, 100) as tracker:
        with drl.tracker.TBMeanTracker(writer, 10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                baseline_buf.add(exp.reward)
                baseline = baseline_buf.mean()

                batch.add(exp)

                reward, step = exp_source.reward_step()
                if reward is not None:
                    tracker.update(reward, step_idx)
                    done_episodes += 1

                if done_episodes > TEST_EPISODES:
                    break

                if len(batch) < BATCH_SIZE:
                    continue

                states, actions, rewards, dones, next_states = batch.unpack()
                batch.clear()

                states_v = torch.FloatTensor(states).to(device)
                actions_v = torch.LongTensor(actions).to(device)
                scales_v = torch.FloatTensor(rewards - baseline).to(device)

                optimizer.zero_grad()
                logits_v = net(states_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                log_prob_actions_v = scales_v * log_prob_v[range(BATCH_SIZE), actions_v]
                loss_policy_v = -log_prob_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_v = - (prob_v * log_prob_v).sum(dim=1).mean()
                loss_entropy_v = - ENTROPY_WEIGHT * entropy_v

                loss_v = loss_policy_v + loss_entropy_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("baseline", baseline, step_idx)
                tb_tracker.track("advantage", scales_v, step_idx)
                tb_tracker.track("loss_entropy", loss_entropy_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("total_loss", loss_v, step_idx)
