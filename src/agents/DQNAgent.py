import torch
from gymnasium import spaces
import numpy as np

from src.agents import agent_utils
from src.agents.Agent import Agent
from src.neural_networks.DuelingFeedForwardNN import DuelingFeedForwardNN
from src.neural_networks.FeedForwardNN import FeedForwardNN
from src.game.FluxxEnums import GameState


class DQNAgent(Agent):
    def __init__(self, game_config, player_number):
        super(DQNAgent, self).__init__()

        self.game_config = game_config
        self.player_number = player_number

        decision_context_length = 19  # same as PPO
        observed_zone_count = 4 + game_config.player_count
        observation_space_size = (
            observed_zone_count * len(game_config.card_list)
            + decision_context_length
            + 2  # draw pile size + opponent hand size
        )
        action_space_size = len(game_config.card_list) + 1

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=1, shape=(observation_space_size,), dtype=np.int8),
            "action_mask": spaces.Box(low=0, high=1, shape=(action_space_size,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(action_space_size)

        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size

        # Q-network. Output dim = number of actions (Q(s, a) for each a).
        self.q_network = FeedForwardNN(observation_space_size, action_space_size, 256)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.q_network.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_device_tensor(self, arr) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device, dtype=torch.float)
        return torch.from_numpy(np.asarray(arr)).float().to(self.device)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = self._to_device_tensor(obs)
        return self.q_network.forward(obs)

    def act(self, state, epsilon: float = 0.0):
        obs = self.encode(state)
        observation = obs["observation"]
        action_mask = obs["action_mask"]

        obs_tensor = self._to_device_tensor(observation)
        mask_tensor = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_values = self.q_network.forward(obs_tensor)
            q_values_masked = q_values.masked_fill(~mask_tensor, float("-inf"))

            if np.random.rand() < epsilon:
                legal_idx = np.flatnonzero(action_mask)
                action = int(np.random.choice(legal_idx))
            else:
                action = int(q_values_masked.argmax(dim=-1).item())

            selected_q = q_values_masked[action].detach().cpu()

        return action, selected_q, obs

    def encode(self, state: GameState):
        if state.game_over:
            dummy_obs = np.zeros(self.observation_space["observation"].shape[0], dtype=np.int8)
            dummy_mask = np.zeros(self.action_space.n, dtype=np.int8)
            return {"observation": dummy_obs, "action_mask": dummy_mask}
        else:
            return agent_utils.observe_hot_encoded(self, state, self.game_config)