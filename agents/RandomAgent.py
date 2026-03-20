from agents.Agent import Agent
import numpy as np

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, obs: dict) -> tuple[int, list[float]]:
        # agent.act() must return log probs for training, but we will never train a RandomAgent, so just return an empty list
        # TODO: return the actual correct type here
        possible_actions = obs["action_mask"]
        return np.random.choice(np.flatnonzero(possible_actions)), []
