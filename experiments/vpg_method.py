import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import drl_lib
import gym
import time
import math

from tensorboardX import SummaryWriter

GAMMA = 1
LEARNING_RATE = 5e-3
ENTROPY_WEIGHT = 0.01
BATCH_SIZE = 64
BASELINE_STEPS = 10000

REWARD_STEPS = 1
GRAD_L2_CLIP = 0.1
TEST_EPISODES = 3000

class LogisticsPGN (nn.Module):
    def __init__ (self, ob_dim, action_dim):
        super(LogisticsPGN, self).__init__ ()

        self.base = nn.Sequential(
                nn.Linear(ob_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
                )

        self.mu = nn.Sequential(
                nn.Linear(64, action_dim),
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = drl_lib.env.supply_chain.SupplyChain(periodic_demand=False)

    net = LogisticsPGN(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    agent = Agent(net, device)

    writer = SummaryWriter(comment='-vpg')
    exp_source = drl_lib.experience.ExperienceSource(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    total_steps = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    batch_states, batch_scales, batch_actions = [], [], []
    m_basline, m_batch_scales, m_loss_policy, m_loss_total, m_entropy_loss = [], [], [], [], []

    baseline_buf = drl_lib.utils.MeanBuffer(BASELINE_STEPS)
    tb_tracker = drl_lib.tracker.TBMeanTracker(writer, 10)
    with drl_lib.tracker.RewardTracker(writer, 100) as tracker:
        for step_idx, exp in enumerate(exp_source):
            baseline_buf.add(exp.reward)
            baseline = baseline_buf.mean()

            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            batch_scales.append(exp.reward)

            reward, step = exp_source.reward_step()
            if reward is not None:
                tracker.update(reward, step_idx)
                done_episodes += 1

            if done_episodes > TEST_EPISODES:
                break

            if len(batch_states) < BATCH_SIZE:
                continue
            states_v = torch.FloatTensor(batch_states).to(device)
            actions_v = torch.FloatTensor(batch_actions).to(device)
            scales_v = torch.FloatTensor(batch_scales).to(device)

            optimizer.zero_grad()
            mu_v, var_v, val_v = net(states_v)

            val_loss_v = F.mse_loss(val_v, scales_v)
            scales_v = scales_v - val_v.detach()

            log_prob_v = calc_logprob(mu_v, var_v, actions_v)
            log_prob_v = scales_v.unsqueeze(-1) * log_prob_v
            loss_policy_v = -log_prob_v.mean()

            entropy_v = (-(torch.log(2 * math.pi * var_v) + 1) / 2).mean()
            entropy_loss_v = ENTROPY_WEIGHT * entropy_v

            loss_v = loss_policy_v + entropy_loss_v + val_loss_v
            loss_v.backward()
            optimizer.step()

            tb_tracker.track("baseline", baseline, step_idx)
            tb_tracker.track("entropy", entropy_v.item(), step_idx)
            tb_tracker.track("scale", scales_v, step_idx)
            tb_tracker.track("entropy_loss", entropy_loss_v.item(), step_idx)
            tb_tracker.track("policy_loss", loss_policy_v.item(), step_idx)
            tb_tracker.track("total_loss", loss_v.item(), step_idx)
            tb_tracker.track("val_loss", val_loss_v.item(), step_idx)
            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    tb_tracker.close()
