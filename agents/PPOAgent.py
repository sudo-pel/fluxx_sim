import torch
from gymnasium import spaces
from torch import nn
import torch.nn.functional as F
import numpy as np

from agents import agent_utils
from agents.Agent import Agent
from agents.FeedForwardNN import FeedForwardNN
from fluxx.game.FluxxEnums import GameState


class PPOAgent(Agent):
    def __init__(self, game_config, player_number):
        super(PPOAgent, self).__init__()

        self.game_config = game_config
        self.player_number = player_number

        decision_context_length = 19 # 7 PLACE zones + play a card + play for opponent, 7 REMAIN zone, 1 int for decisions left, 1 int for counter, 1 int for on_complete [draw]
        observed_zone_count = 4 + game_config.player_count # hand (for observing agent), goals, rules, keepers, discard pile (for each agent)
        observation_space_size = observed_zone_count * len(game_config.card_list) + decision_context_length + 2 # +2 for draw pile size and opponent hand size

        action_space_size = len(game_config.card_list) + 1 # +1 for "don't use a free action"

        self.observation_space = spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(observation_space_size,), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(action_space_size,), dtype=np.int8),
        })

        self.action_space = spaces.Discrete(action_space_size)

        in_dim = observation_space_size
        out_dim = action_space_size

        self.policy_network = FeedForwardNN(in_dim, out_dim)

    def forward(self, obs):
        # Convert observation to tensor if given as a numpy array
        return self.policy_network.forward(obs)

    def act(self, state):
        obs = self.encode(state)

        observation = obs["observation"]
        action_mask = obs["action_mask"]

        logits = self.policy_network.forward(observation)
        logits[action_mask == 0] = -float("inf")

        distribution = torch.distributions.Categorical(logits=logits)

        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        return action.item(), log_probs, obs

    def encode(self, state: GameState):
        if state.game_over:
            dummy_obs = np.zeros(self.observation_space["observation"].shape[0], dtype=np.int8)
            dummy_mask = np.zeros(self.action_space.n, dtype=np.int8)
            return {
                "observation": dummy_obs,
                "action_mask": dummy_mask,
            }
        else:
            return agent_utils.observe(self, state, self.game_config)

