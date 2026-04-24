# src/neural_networks/DuelingFeedForwardNN.py

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DuelingFeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super(DuelingFeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.value_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.advantage_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.advantage_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs, action_mask=None):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        h1 = F.relu(self.layer1(obs))
        h2 = F.relu(self.layer2(h1))

        v = F.relu(self.value_hidden(h2))
        v = self.value_head(v)  # (..., 1)

        a = F.relu(self.advantage_hidden(h2))
        a = self.advantage_head(a)  # (..., out_dim)

        if action_mask is not None:
            # Accept numpy masks too, for symmetry with obs handling
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.tensor(action_mask, dtype=torch.bool, device=a.device)
            legal = action_mask.to(dtype=a.dtype)
            legal_count = legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
            a_mean = (a * legal).sum(dim=-1, keepdim=True) / legal_count
        else:
            a_mean = a.mean(dim=-1, keepdim=True)

        q = v + (a - a_mean)
        return q