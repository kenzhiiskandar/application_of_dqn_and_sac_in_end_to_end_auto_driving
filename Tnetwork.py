#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:03:18 2023

@author: oscar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from vit_backbone import SimpleViT

if torch.cuda.is_available():
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print('Use:', device)

##############################################################################
class Net(nn.Module):

    def __init__(self, height, width, channel, num_outputs, seed):
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
            channels = channel
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
        x = F.softmax(x, dim=1)
        return x
    
class Net2(nn.Module):

    def __init__(self, height, width, channel, num_outputs, seed):
        super(Net2, self).__init__()
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
            channels = channel
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
