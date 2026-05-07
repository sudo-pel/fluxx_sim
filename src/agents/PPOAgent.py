import torch
from gymnasium import spaces
import numpy as np

from src.agents import agent_utils
from src.agents.Agent import Agent
from src.neural_networks.FeedForwardNN import FeedForwardNN
from src.game.FluxxEnums import GameState


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

    @property
    def device(self) -> torch.device:
        """
        Device of the underlying policy network. Inferred from parameters so
        callers don't have to plumb a device through the constructor.
        """
        try:
            return next(self.policy_network.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_device_tensor(self, arr) -> torch.Tensor:
        """
        Convert a numpy array (or tensor) observation to a float tensor on the
        policy network's device.
        """
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device, dtype=torch.float)
        return torch.from_numpy(np.asarray(arr)).float().to(self.device)

    def forward(self, obs):
        # Convert observation to tensor if given as a numpy array
        if isinstance(obs, np.ndarray):
            obs = self._to_device_tensor(obs)
        return self.policy_network.forward(obs)

    def act(self, state):
        obs = self.encode(state)

        observation = obs["observation"]
        action_mask = obs["action_mask"]

        # Move observation + mask onto the network's device for the forward pass.
        obs_tensor = self._to_device_tensor(observation)
        mask_tensor = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits = self.policy_network.forward(obs_tensor)
            # Out-of-place mask fill is safer than in-place assignment on device tensors.
            logits = logits.masked_fill(~mask_tensor, float("-inf"))

            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_probs = distribution.log_prob(action)

        # Return a python int for the action (env.step expects a plain int) and a
        # detached CPU scalar tensor for log_probs so the training loop can stack
        # them without device-mismatch issues. `obs` is kept as numpy (unchanged
        # contract) — the training loop is responsible for batching it.
        return action.item(), log_probs.detach().cpu(), obs

    def encode(self, state: GameState):
        if state.game_over:
            dummy_obs = np.zeros(self.observation_space["observation"].shape[0], dtype=np.int8)
            dummy_mask = np.zeros(self.action_space.n, dtype=np.int8)
            return {
                "observation": dummy_obs,
                "action_mask": dummy_mask,
            }
        else:
            return agent_utils.observe_hot_encoded(self, state, self.game_config)