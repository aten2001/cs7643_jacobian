from __future__ import division
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class MLP(nn.Module):
    '''
    Simple MLP to demonstrate Jacobian regularization.
    '''
    def __init__(self, in_channel=1, im_size=28, num_classes=10, 
                 fc_channel1=200, fc_channel2=200):
        super(MLP, self).__init__()
        
        # Parameter setup
        compression=in_channel*im_size*im_size
        self.compression=compression
        
        # Structure
        self.fc1 = nn.Linear(compression, fc_channel1)
        self.fc2 = nn.Linear(fc_channel1, fc_channel2)
        self.fc3 = nn.Linear(fc_channel2, num_classes)
        
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = x.view(-1, self.compression)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
