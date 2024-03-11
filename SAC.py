#!/usr/bin/env python3

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_value_
from torch.distributions.categorical import Categorical

from Tnetwork import Net, Net2, device
from cpprb import PrioritizedReplayBuffer

class SAC():
    
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
        self.actor_lr = 1e-3
        self.critic_lr = 1e-2
        self.alpha_lr = 1e-2
        self.target_entropy = -1
        
        self.lr_p = 0.0001
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

        # Policy Network
        self.actor_net = Net(height, width, channel, n_actions, seed)

        # Q Network 1 and 2
        self.critic_1_net = Net2(height, width, channel, n_actions, seed)
        self.critic_2_net = Net2(height, width, channel, n_actions, seed)

        # Target Q Network 1 and 2
        self.target_critic_1_net = Net2(height, width, channel, n_actions, seed)
        self.target_critic_2_net = Net2(height, width, channel, n_actions, seed)

        # GPU Matters
        if torch.cuda.device_count() > 8:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.actor_net = nn.DataParallel(self.actor_net)
            self.critic_1_net = nn.DataParallel(self.critic_1_net)
            self.critic_2_net = nn.DataParallel(self.critic_2_net)
            self.target_critic_1_net = nn.DataParallel(self.target_critic_1_net)
            self.target_critic_2_net = nn.DataParallel(self.target_critic_2_net)

            self.actor_net.to(device)
            self.critic_1_net.to(device)
            self.critic_2_net.to(device)
            self.target_critic_1_net.to(device)
            self.target_critic_2_net.to(device)
        else:
            self.actor_net.to(device)
            self.critic_1_net.to(device)
            self.critic_2_net.to(device)
            self.target_critic_1_net.to(device)
            self.target_critic_2_net.to(device)

        # Let the initial parameters of the target Q network be the same as the Q network
        self.target_critic_1_net.load_state_dict(self.critic_1_net.state_dict())
        self.target_critic_2_net.load_state_dict(self.critic_2_net.state_dict())
        
        ##### Loss and Optimizer ###
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(),lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1_net.parameters(),lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2_net.parameters(),lr=self.critic_lr)
        
        # Using the log value of alpha can make the training results more stable.
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # You can find the gradient of alpha   
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.alpha_lr)

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
        # Select action randomly based on the probabilities predicted by the actor policy net
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None].copy()) / 255.0
            probs = self.actor_net.forward(x)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action_dist, \
                torch.tensor([action.item()], device=device, dtype=torch.long)
    
    def select_action_deterministic(self, x):
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None]) / 255.0
            probs = self.actor_net.forward(x)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return 1.0, \
                torch.tensor([action.item()], device=device, dtype=torch.long)



    def calc_td_target(self, rewards, next_states, dones):
        next_probs = self.actor_net(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1_net(next_states)
        q2_value = self.target_critic_2_net(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def store_transition(self, s, a, r, s_, engage, d=0):
        self.replay_buffer.add(obs=s,
                act=a,
                rew=r,
                next_obs=s_,
                engage = engage,
                done=d)
        
    def optimize_network(self):

        # Sample Batch 
        data = self.replay_buffer.sample(self.batch_size)
        istates, actions, engages = data['obs'], data['act'], data['engage']
        rewards, next_istates, dones = data['rew'], data['next_obs'], data['done']

        state_batch = torch.FloatTensor(istates).permute(0,3,1,2).to(device) / 255.0
        action_batch = torch.FloatTensor(actions).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        engages = torch.FloatTensor(engages).to(device)
        next_state_batch = torch.FloatTensor(next_istates).permute(0,3,1,2).to(device) / 255.0
        dones = torch.FloatTensor(dones).to(device)

        # Update two Q networks
        td_target = self.calc_td_target(reward_batch, next_state_batch, dones)

        critic_1_q_values = self.critic_1_net(state_batch).gather(1, action_batch.type(torch.int64))
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))

        critic_2_q_values = self.critic_2_net(state_batch).gather(1, action_batch.type(torch.int64))
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update policy networks
        probs = self.actor_net(state_batch)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #Calculate entropy directly based on probability
        q1_value = self.critic_1_net(state_batch)
        q2_value = self.critic_2_net(state_batch)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # Calculate expectations directly based on probability  
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha value
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1_net, self.target_critic_1_net)
        self.soft_update(self.critic_2_net, self.target_critic_2_net)
    