#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:08:43 2023

@author: oscar
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_value_
from torch.distributions.categorical import Categorical

from Tnetwork import Net, device
from cpprb import PrioritizedReplayBuffer

class DQN():
    
    def __init__(self,
                 height,
                 width,
                 channel,
                 n_obs,
                 n_actions,
                 IMPORTANTSAMPLING,
                 ENTROPY,
                 BATCH_SIZE,
                 GAMMA,
                 EPS_START,
                 EPS_END,
                 EPS_EPOC,
                 THRESHOLD,
                 MEMORY_CAPACITY,
                 seed,
                 ):
        self.height = height
        self.width = width
        self.channel = channel
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_epoc = EPS_EPOC
        self.threshold = THRESHOLD
        self.memory_capacity = MEMORY_CAPACITY
        self.seed = seed
        self.auto_entropy = ENTROPY
        self.imsamp = IMPORTANTSAMPLING
        
        ##### Hyper Parameters ####
        self.lr = 0.00025
        self.lr_p = 0.0001
        self.lr_temp = 0.001
        self.alpha = 0.95
        self.eps = 0.01
        self.tau = 0.005

        self.steps_done = 0
        self.loss_critic = 0.0
        self.loss_engage_q = 0.0
        self.eps_threshold = 0.0
        self.q = 0.0
        self.target_entropy_ratio = 0.3

        self.q_policy = np.zeros(self.n_actions)
        self.q_target = np.zeros(self.n_actions)

        ##### Fix Seed ####
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        ##### Initializing Net ####
        self.policy_net = Net(height, width, channel, n_actions, seed)#.to(device)
        self.target_net = Net(height, width, channel, n_actions, seed)#.to(device)

        if torch.cuda.device_count() > 8:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.policy_net = nn.DataParallel(self.policy_net)
            self.target_net = nn.DataParallel(self.target_net)
            self.policy_net.to(device)
            self.target_net.to(device)
        else:
            self.policy_net.to(device)
            self.target_net.to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        ##### Loss and Optimizer ####
        self.SL = nn.SmoothL1Loss()
        self.normalize = nn.Softmax(dim=1)
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=self.lr,
                                       alpha=self.alpha, eps=self.eps)
             
        ##### Temperature Adjustment ####
        if self.auto_entropy:
            self.target_entropy = \
                -np.log(1.0 / self.n_actions) * self.target_entropy_ratio
            self.log_temp = torch.zeros(1, requires_grad=True, device=device)
            self.temperature = self.log_temp.exp()
            self.temp_optim = optim.Adam([self.log_temp], lr=self.lr_temp)
        else:
            self.temperature = 0.2 #1.0 / max(1, self.n_actions)

        ##### Replay Buffer ####
        self.replay_buffer = PrioritizedReplayBuffer(self.memory_capacity,
                                          {"obs": {"shape": (self.height,self.width,9),"dtype": np.uint8},
                                           "act": {},
                                           "rew": {},
                                           "next_obs": {"shape": (self.height,self.width,9),"dtype": np.uint8},
                                           "engage": {},
                                           "done": {}},
                                          next_of=("obs"))

    def select_action(self, x, i_epoc):
        sample = random.random()
        
        self.eps_threshold = self.calc_eps_threshold(i_epoc)
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None].copy()) / 255.0
        
            ##### Greedy Action ####
            if sample > self.eps_threshold:    
                q = self.policy_net.forward(x)
                self.q = q.squeeze(0).cpu().numpy()
                q_distribution = self.normalize(q)
                return q_distribution.squeeze(0).cpu().numpy(), q.max(1)[1].view(1, 1)
            
            ##### Stochastic Action #### 
            else:
                q = self.policy_net.forward(x)
                self.q = q.squeeze(0).cpu().numpy()
                q_distribution = self.normalize(q)
                return q_distribution.squeeze(0).cpu().numpy(), \
                       torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def select_action_deterministic(self, x):
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None]) / 255.0
            q = self.policy_net.forward(x)
            return 1.0, q.max(1)[1].view(1, 1)

    def calc_eps_threshold(self, i_epoc):
        if (i_epoc <= self.threshold):
            return self.eps_start
        else:
            fraction = min((i_epoc - self.threshold) / self.eps_epoc, 1.0)
            return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def optimize_network(self):

        ##### Sample Batch #####
        data = self.replay_buffer.sample(self.batch_size)
        istates, actions, engages = data['obs'], data['act'], data['engage']
        rewards, next_istates, dones = data['rew'], data['next_obs'], data['done']

        state_batch = torch.FloatTensor(istates).permute(0,3,1,2).to(device) / 255.0
        action_batch = torch.FloatTensor(actions).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        engages = torch.FloatTensor(engages).to(device)
        next_state_batch = torch.FloatTensor(next_istates).permute(0,3,1,2).to(device) / 255.0
        dones = torch.FloatTensor(dones).to(device)

        ##### Q value #####
        q_policy = self.policy_net.forward(state_batch)
        q_policy_selected = q_policy.gather(1, action_batch.type(torch.int64))
        next_q_target = torch.zeros(self.batch_size, device=device)
        next_q_target = self.target_net.forward(next_state_batch).max(1)[0].detach()

        ##### Q target ####
        q_target = (next_q_target.unsqueeze(1) * self.gamma) + reward_batch

        ##### critic loss ######
        loss_critic = self.SL(q_policy_selected, q_target)
        if torch.isnan(loss_critic):
            print('q loss is nan.')
        self.loss_critic = loss_critic.detach().cpu().numpy()

        loss_engage = 0.0
        self.loss_engage_q = loss_engage

        ##### Overall Loss #####
        self.optimizer.zero_grad()
        loss_total = loss_critic + loss_engage

        ##### Optimization #####
        loss_total.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        ##### Soft Update Target Network #####
        self.soft_update(self.target_net, self.policy_net, self.tau)

    def store_transition(self, s, a, r, s_, engage, d=0):
        self.replay_buffer.add(obs=s,
                act=a,
                rew=r,
                next_obs=s_,
                engage = engage,
                done=d)

    def optimize_entropy_parameter(self, entropy):
        temp_loss = -torch.mean(self.log_temp * (self.target_entropy + entropy))
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        self.temperature = self.log_temp.detach().exp()
        
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    