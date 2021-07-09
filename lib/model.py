from gym.core import Env
import numpy as np
import drl

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 256

class ThresholdAgent:
    def __init__ (self, eps: np.array, Q: np.array, action_dim: int):
        self.eps = eps
        self.Q = Q
        self.action_dim = action_dim
        self.production_flag = True

    def get_action (self, state: np.ndarray):
        action = np.zeros(self.action_dim, dtype=np.int32)
        action[0] = self.Q[0] if self.production_flag else 0
        for i in range(1, self.action_dim):
            if state[i] < self.eps[i]:
                action[i] = self.Q[i]

        return action

    def set_production_level (self, state: np.ndarray, num_storages: int):
        self.production_flag = False
        if (state[0] - np.sum(state[1:num_storages+1])) < self.eps[0]:
            self.production_flag = True

class A2CModel (nn.Module):
    def __init__ (self, obs_size, act_size):
        super(A2CModel, self).__init__()

        self.base = nn.Sequential(
                nn.Linear(obs_size, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                )

        self.mu = nn.Sequential(
                nn.Linear(HID_SIZE, act_size),
                # nn.ReLU(),
                )

        self.val = nn.Sequential(
            nn.Linear(HID_SIZE, 1)
        )

        self.logstd = nn.Parameter(torch.zeros((1, act_size)))

    def forward (self, x):
        base_out = self.base(x)
        mu = self.mu(base_out)
        val = self.val(base_out)

        return mu, val


class MulSpActorModel (nn.Module):
    def __init__ (self, input_shape, act_size):
        super(MulSpActorModel, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
                )

        out_size = drl.common.utils.get_conv_out(self.conv, input_shape)
        self.mu = nn.Sequential(
                nn.Linear(out_size, 64),
                nn.ReLU(),
                nn.Linear(64, act_size)
                )

        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(x.size()[0], -1)
        mu = self.mu(conv_out)
        return mu



class MatrixModel (nn.Module):
    def __init__ (self, input_shape, act_size):
        super(MatrixModel, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv1d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                )
        conv_out_size = drl.common.utils.get_conv_out(self.conv, input_shape)
        self.mu = nn.Sequential(
                nn.Linear(conv_out_size, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, act_size),
                # nn.ReLU(),
                )

        self.val = nn.Sequential(
                nn.Linear(conv_out_size, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, 1),
                )

        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward (self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(x.size()[0], -1)
        mu = self.mu(conv_out)
        val = self.val(conv_out)

        return mu, val

class MatrixModel2 (nn.Module):
    def __init__ (self, obs_size, act_size):
        super(MatrixModel2, self).__init__()

        self.layer1 = nn.Linear(obs_size, 12)
        self.layer2 = nn.Linear(obs_size, 12)
        self.layer3 = nn.Linear(obs_size, 12)

        self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                )
        conv_out_size = drl.common.utils.get_conv_out(self.conv, (3, 3, 4))
        self.mu = nn.Sequential(
                nn.Linear(conv_out_size, HID_SIZE),
                nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                nn.Linear(HID_SIZE, act_size),
                # nn.ReLU(),
                )

        # self.val = nn.Sequential(
        #         nn.Linear(conv_out_size, HID_SIZE),
        #         nn.ReLU(),
        #         nn.Linear(HID_SIZE, 1),
        #         )

        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward (self, x):
        fx = x.float()
        layer1 = self.layer1(fx)
        layer2 = self.layer2(fx)
        layer3 = self.layer3(fx)
        conv_in = torch.cat([layer1, layer2, layer3], 1).view(x.size()[0], 3, 3, 4)

        conv_out = self.conv(conv_in).view(x.size()[0], -1)
        mu = self.mu(conv_out)
        # val = self.val(conv_out)

        return mu, 0


class A2CAgent (drl.agent.BaseAgent):
    def __init__ (self, net, env=None, device="cpu"):
        self.net = net
        self.device = device
        self.env = env

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states).to(self.device)
        states = np.array(states, copy=False)

        mu_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        
        # for idx, action in enumerate(actions):
        #     actions[idx] = self.env.clipping_action(action)

        return actions, agent_states

class NormalAgent (drl.agent.BaseAgent):
    def __init__ (self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor([states]).to(self.device)
        states = np.array(states, copy=False)
        mu_v, _ = self.net(states_v)
        actions = mu_v.data.cpu().numpy()[0]
        
        
        # for idx, action in enumerate(actions):
        #     actions[idx] = self.env.clipping_action(action)

        return actions, agent_states

class A2CAgent2 (drl.agent.BaseAgent):
    def __init__ (self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states).to(self.device)

        mu_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd

        return actions, agent_states


class NoisyA2CAgent (drl.agent.BaseAgent):
    def __init__ (self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states).to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()

        return mu, agent_states


class DDPGActor (nn.Module):
    def __init__ (self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(obs_size, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                nn.Linear(HID_SIZE, act_size),
                # nn.ReLU(),
                )

    def forward (self, x):
        fx = x.float()
        return self.net(fx)

class DDPGCritic (nn.Module):
    def __init__ (self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
                nn.Linear(obs_size, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                # nn.Linear(HID_SIZE, HID_SIZE),
                # nn.ReLU(),
                )

        self.out_net = nn.Sequential(
                nn.Linear(HID_SIZE + act_size, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, 1)
                )

    def forward (self, x, a):
        fx, fa = x.float(), a.float()
        obs = self.obs_net(fx)
        return self.out_net(torch.cat([obs, fa], dim=1))

class AgentDDPG(drl.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, storage_capacity, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.5, ou_sigma=0.6,
                 ou_epsilon=1.0):
        self.storage_capacity = storage_capacity
        self.num_stores = storage_capacity.shape[0]
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(
                    size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        return actions, new_a_states
