import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        # Convert observation to tensor if given as a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

    def act(self, state):
        observation = state["observation"]
        action_mask = state["action_mask"]

        logits = self.forward(observation)
        logits[action_mask == 0] = -float("inf")

        distribution = torch.distributions.Categorical(logits=logits)

        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        return action.item(), log_probs
