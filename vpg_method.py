import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import drl_lib
import gym
import time
import math
import argparse
import ptan

from tensorboardX import SummaryWriter

GAMMA = 1
LEARNING_RATE = 1e-4
ENTROPY_WEIGHT = 0.01
BATCH_SIZE = 16
BASELINE_STEPS = 1000

REWARD_STEPS = 1
GRAD_L2_CLIP = 0.1
TEST_EPISODES = 10000

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

        self.mu = nn.Sequential(
                nn.Linear(64, action_dim),
                )

        self.var = nn.Sequential(
                nn.Linear(64, action_dim),
                nn.Softplus()
                )

        self.val = nn.Sequential(
                nn.Linear(64, 1)
                )

        w = torch.zeros(action_dim, dtype=torch.float32)
        self.log_std = nn.Parameter(w)

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
        out_mean = self.mu(out_base)
        out_val = self.val(out_base)
        out_var = self.var(out_base)

        return out_mean, F.softplus(self.log_std), out_val


class Agent (drl_lib.agent.BaseAgent):
    def __init__ (self, model: nn.Module, device="cpu", preprocessor=drl_lib.utils.Preprocessor.default_tensor):
        super(Agent, self).__init__()
        self.model = model
        self.device = device
        self.preprocessor = preprocessor

    def __call__ (self, state: np.ndarray):
        if self.preprocessor is not None:
            state = self.preprocessor(state)
        if torch.is_tensor(state):
            state = state.to(self.device)

        mu_v, var_v, _ = self.model(state)
        mu = mu_v.squeeze(0).data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()

        action = np.random.normal(mu, sigma)
        return action
 
def calc_logprob (mu_v: torch.Tensor, var_v: torch.Tensor, action_v: torch.Tensor) -> torch.Tensor:
    p1 = - ((mu_v - action_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v.clamp(min=1e-3)))

    return p1 + p2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-sd", action='store_true', default=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = drl_lib.env.supply_chain.SupplyChain()
    if args.sd == True:
        print("supply distribution 10")
        env = drl_lib.env.supply_distribution10.SupplyDistribution(
                n_stores=3, cap_truck=2, prod_cost=1, max_prod=3,
                store_cost=np.array([0, 2, 0, 0]),
                truck_cost=np.array([3, 3, 0]),
                cap_store=np.array([50, 10, 10, 10]),
                penalty_cost=1, price=2.5, gamma=1, max_demand=3, episode_length=25)

    net = LogisticsPGN(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    agent = Agent(net, device)

    writer = SummaryWriter(comment=f'-cont_vpg-{args.name}')
    exp_source = drl_lib.experience.ExperienceSource(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    done_episodes = 0

    baseline_buf = drl_lib.utils.MeanBuffer(BASELINE_STEPS)
    batch = drl_lib.experience.BatchData(max_size=BATCH_SIZE)
    with drl_lib.tracker.RewardTracker(writer, 100) as tracker:
        with drl_lib.tracker.TBMeanTracker(writer, 10) as tb_tracker:
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
                actions_v = torch.FloatTensor(actions).to(device)
                scales_v = torch.FloatTensor(rewards - baseline).to(device)

                optimizer.zero_grad()
                mu_v, var_v, val_v = net(states_v)

                log_prob_v = calc_logprob(mu_v, var_v, actions_v)
                log_prob_v = scales_v.unsqueeze(-1) * log_prob_v
                loss_policy_v = -log_prob_v.mean()

                entropy_v = (-(torch.log(2 * math.pi * var_v) + 1) / 2).mean()
                entropy_loss_v = ENTROPY_WEIGHT * entropy_v

                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("baseline", baseline, step_idx)
                tb_tracker.track("entropy", entropy_v, step_idx)
                tb_tracker.track("advantage", scales_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss", loss_v, step_idx)

