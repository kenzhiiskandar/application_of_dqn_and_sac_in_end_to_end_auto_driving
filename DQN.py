import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_value_
from torch.distributions.categorical import Categorical

from cpprb import PrioritizedReplayBuffer

from einops import rearrange
from einops.layers.torch import Rearrange

if torch.cuda.is_available():
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print('Use:', device)

####################################################################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.attn = None

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        print("SIMPEVIT, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64)", image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.layer_norm = nn.LayerNorm(dim)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype
        #print("img", img)
        #print("img.shape", img.shape)
        x = self.to_patch_embedding(img)
        #print("self.to_patch_embedding(img)", self.to_patch_embedding(img))
        #print("self.to_patch_embedding(img).shape", self.to_patch_embedding(img).shape)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)
        x = self.layer_norm(x)
        return x


##############################################################################
class Net(nn.Module):

    def __init__(self, height, width, num_outputs, seed):
        super(Net, self).__init__()
        self.height = height
        self.width = width
        self.linear_dim = 128
        self.hidden_dim = 512
        self.feature_dim = 256
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.trans = SimpleViT(
            image_size = (self.height, self.width ),
            patch_size = (int(self.height/5), int(self.width/5)),
            num_classes = 2,
            dim = self.feature_dim,
            depth = 2,
            heads = 8,
            mlp_dim = self.hidden_dim,
            channels = 1
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, self.linear_dim),
            nn.LayerNorm(self.linear_dim),
            nn.ReLU(),
            nn.Linear(self.linear_dim, num_outputs)
            )

    def forward(self, x):
        x = x.to(device)
        x = self.trans(x)
        x = x.contiguous().view(-1, self.feature_dim)
        x = self.fc(x)
        return x
        
##############################################################################


class DQN():
    
    def __init__(self,
                 height,
                 width,
                 n_obs,
                 n_actions,
                 BATCH_SIZE,
                 GAMMA,
                 EPS_START,
                 EPS_END,
                 EPS_EPOC,
                 seed,
                 ):
        self.height = height
        self.width = width
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_epoc = EPS_EPOC
        self.seed = seed
         
        ##### Hyper Parameters ####
        self.lr = 0.00025
        self.lr_p = 0.0001
        self.lr_temp = 0.001
        self.alpha = 0.95
        self.eps = 0.01
        self.tau = 0.005
        self.steps_done = 0
        self.loss_critic = 0.0
        self.loss_actor = 0.0
        self.loss_entropy = 0.0
        self.loss_engage_q = 0.0
        self.loss_engage_preference = 0.0
        self.eps_threshold = 0.0
        self.q = 0.0
        self.action_distribution = 0.0
        self.target_entropy_ratio = 0.3
        self.temperature_copy = 0.0
        self.default_weight = np.exp(1.0)
        self.policy_guidance = False
        self.value_guidance = False
        self.adaptive_weight = False
        self.ac_dis_policy = np.zeros(self.n_actions)
        self.ac_dis_target = np.zeros(self.n_actions)
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
        self.policy_net = Net(height, width, n_actions, seed).to(device)
        self.target_net = Net(height, width, n_actions, seed).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        ##### Loss and Optimizer ####
        self.GL = nn.GaussianNLLLoss(reduction='none')
        self.KL = nn.KLDivLoss()
        self.CE = nn.CrossEntropyLoss()
        self.SL = nn.SmoothL1Loss()
        self.normalize = nn.Softmax(dim=1)
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=self.lr,
                                       alpha=self.alpha, eps=self.eps)
        self.optimizer_p = optim.RMSprop(self.policy_net.parameters(),lr=self.
                                         lr_p, alpha=self.alpha, eps=self.eps)

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)
        self.scheduler_p = lr_scheduler.ExponentialLR(self.optimizer_p, gamma=1.0)

        ##### Temperature Adjustment ####
        '''
        if self.auto_entropy:
            self.target_entropy = \
                -np.log(1.0 / self.n_actions) * self.target_entropy_ratio
            self.log_temp = torch.zeros(1, requires_grad=True, device=device)
            self.temperature = self.log_temp.exp()
            self.temp_optim = optim.Adam([self.log_temp], lr=self.lr_temp)
        else:
        '''
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
                
                if (self.preference):
                    action_distribution, q = self.policy_net.forward(x)
                    action_idx = q.max(1)[1].view(1, 1)
                    self.action_distribution = action_distribution.squeeze(0).cpu().numpy()
                    self.q = q.squeeze(0).cpu().numpy()
                    return self.action_distribution, action_idx
                
                q = self.policy_net.forward(x)
                self.q = q.squeeze(0).cpu().numpy()
                q_distribution = self.normalize(q)
                return q_distribution.squeeze(0).cpu().numpy(), q.max(1)[1].view(1, 1)
            
            ##### Stochastic Action ####
            else:
                if (self.preference):
                    
                    action_distribution, q = self.policy_net.forward(x)
                    self.q = q.squeeze(0).cpu().numpy()
                    action_distribution = action_distribution.squeeze(0).cpu().numpy()
                    distribution = action_distribution / action_distribution.sum().tolist()
                    distribution = np.nan_to_num(distribution)
                    
                    return action_distribution, \
                           torch.tensor([[np.random.choice(np.arange(0, self.n_actions),\
                                        p=distribution)]], device=device, dtype=torch.long)

                q = self.policy_net.forward(x)
                self.q = q.squeeze(0).cpu().numpy()
                q_distribution = self.normalize(q)
                return q_distribution.squeeze(0).cpu().numpy(), \
                       torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_Q_network(self, state, action, reward, next_state, driver_model,
                       variance_list):

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
        if (not self.preference):
            q_policy = self.policy_net.forward(state_batch)
            q_policy_selected = q_policy.gather(1, action_batch.type(torch.int64))
        else:
            action_distribution, q_policy = self.policy_net.forward(state_batch)
            q_policy_selected = q_policy.gather(1, action_batch.long())

        next_q_target = torch.zeros(self.batch_size, device=device)

        next_q_target = self.target_net.forward(next_state_batch).max(1)[0].detach()

        ##### Q target ####
        q_target = (next_q_target.unsqueeze(1) * self.gamma) + reward_batch

        ##### critic loss ######
        loss_critic = self.SL(q_policy_selected, q_target)
        if torch.isnan(loss_critic):
            print('q loss is nan.')
        self.loss_critic = loss_critic.detach().cpu().numpy()

        ##### engage loss ######
        
        ##### UnHiL, HIRL, EIL ######
        if self.value_guidance:
            engage_index = (engages == 1).nonzero(as_tuple=True)[0]
            if engage_index.numel() > 0:
                states_expert = state_batch[engage_index]
                actions_expert = action_batch[engage_index]
                actions_rl = q_policy[engage_index]
                one_hot_expert_actions = torch.squeeze(F.one_hot(actions_expert.long(),
                                                                 self.n_actions), axis=1)
                expected_var = torch.ones_like(actions_rl)
                
                driver_mean, driver_variance, probability, driver_a =\
                    self.driver_decision(states_expert, driver_model, online=False)
                    
                var_max = max(variance_list)
                var_min = min(variance_list)
                var_selected = driver_variance[np.arange(actions_expert.size(dim=0)),
                                               torch.squeeze(actions_expert).long()]
                x = (var_selected - var_min) / (var_max - var_min + 1e-7)
                engage_weight  = torch.exp(-2 * x + 1)
                
                #####!!!!!! Adaptive Confidence Adjustment !!!!!!#####
                if self.adaptive_weight:
                    loss_engage = (self.GL(actions_rl, one_hot_expert_actions,
                                            expected_var).mean(dim=1) * engage_weight).mean()
                else:
                    loss_engage = (self.GL(actions_rl, one_hot_expert_actions,
                                            expected_var).mean(dim=1)).mean() * self.default_weight

                self.loss_engage_q = loss_engage.detach().cpu().numpy()
            else:
                loss_engage = 0.0
                self.loss_engage_q = loss_engage
                
        ###### IARL, DRL ######
        else:
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

    def policy_gradient(self, state, action, reward, next_state,
                        engage, driver_mean, driver_variance, variance_list):
        
        state_tensor = torch.FloatTensor(state.transpose(2,0,1)[None].copy()).to(device) / 255.0
        next_state_tensor = torch.FloatTensor(state.transpose(2,0,1)[None].copy()).to(device) / 255.0
        action_distribution_policy, q_policy = self.policy_net.forward(state_tensor)
        action_distribution_target, _ = self.target_net.forward(state_tensor)
        _, next_q_target_temp = self.target_net.forward(next_state_tensor)
        q_target = torch.from_numpy(np.array(reward)).to(device) +\
                   next_q_target_temp * self.gamma

        ##### For main function #####
        self.ac_dis_policy = action_distribution_policy.squeeze(0).cpu().detach().numpy()
        self.ac_dis_target = action_distribution_target.squeeze(0).cpu().detach().numpy()
        self.q_policy = q_policy.squeeze(0).cpu().detach().numpy()
        self.q_target = q_target.squeeze(0).cpu().detach().numpy()

        action_distribution_policy = action_distribution_policy.squeeze(0)
        action_distribution_target = action_distribution_target.squeeze(0)
        action_prob_policy = Categorical(action_distribution_policy)

        q_policy = q_policy.squeeze(0)
        q_target = q_target.squeeze(0)

        state_value = torch.matmul(action_distribution_target, q_target)
        advantage_function = (q_target - state_value).detach()

        ###### Loss Function ######
        loss_policy = - torch.matmul(action_prob_policy.probs, advantage_function)
        if torch.isnan(loss_policy):
            print('policy loss is nan.')
        self.loss_policy = loss_policy.detach().cpu().numpy()

        loss_entropy =  - action_prob_policy.entropy().mean()
        if torch.isnan(loss_entropy):
            print('entropy loss is nan.')
        self.loss_entropy = loss_entropy.detach().cpu().numpy()

        if self.policy_guidance and engage:
            loss_engage = self.KL(driver_mean.log(), action_distribution_policy)
            self.loss_engage_preference = loss_engage.detach().cpu().numpy()
            var_max = max(variance_list)
            var_min = min(variance_list)
            prob, ind = torch.max(driver_mean, axis=0)
            var_selected = driver_variance[ind]
            x = (var_selected - var_min) / (var_max - var_min + 1e-7)
            engage_weight  = torch.exp(-2 * x + 1)
            
            #####!!!!!! Adaptive Confidence Adjustment !!!!!!#####
            if self.adaptive_weight:
                loss_policy = loss_policy + loss_engage * engage_weight
            else:
                loss_policy = loss_policy + loss_engage * self.default_weight
            
        elif (self.temperature > 0):
            loss_policy = loss_policy + loss_entropy * self.temperature
        else:
            loss_policy = loss_policy

        ##### Temperature Adjustment ######
        if self.auto_entropy:
            self.optimize_entropy_parameter(loss_entropy.detach())
            self.temperature_copy = self.temperature.detach().squeeze().cpu().numpy()
        else:
            self.temperature_copy = self.temperature

        return loss_policy
        
    def optimize_preference_network(self, state, action, reward, next_state, engage,
                               driver_mean, driver_variance, variance_list):

        ##### Something Wrong #####
        if (not self.preference or self.together):
            print(self.preference, '|', self.together, '|RETURN!')
            return

        loss_policy = self.policy_gradient(state, action, reward, next_state,
                                           engage, driver_mean, driver_variance,
                                           variance_list)
        
        ##### Optimization #####
        self.optimizer_p.zero_grad()
        loss_policy.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer_p.step()

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
    