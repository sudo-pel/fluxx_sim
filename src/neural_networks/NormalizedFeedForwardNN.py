import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NormalizedFeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(NormalizedFeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs):
        # Convert observation to tensor if given as a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.norm1(self.layer1(obs)))
        activation2 = F.relu(self.norm2(self.layer2(activation1)))
        output = self.layer3(activation2)

        return output