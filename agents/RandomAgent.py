from agents.Agent import Agent
import numpy as np

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, obs: dict) -> int:
        possible_actions = obs["action_mask"]
        return np.random.choice(np.flatnonzero(possible_actions))
