import sys
sys.path.append('..')
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Discriminator(nn.Module):
    
    def __init__(self, hidden_size=230, num_labels=2):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc1 = nn.Linear(hidden_size, 100)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits

