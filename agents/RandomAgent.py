from gymnasium import spaces

from agents import agent_utils
from agents.Agent import Agent
import numpy as np

from fluxx.game.FluxxEnums import GameState, GameConfig


class RandomAgent(Agent):
    def __init__(self, game_config: GameConfig, player_number: int):
        super().__init__()
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

    def act(self, state: GameState) -> tuple[int, list[float], dict[str, np.ndarray]]:
        obs = self.encode(state)

        # agent.act() must return log probs for training, but we will never train a RandomAgent, so just return an empty list
        # TODO: return the actual correct type here
        possible_actions = obs["action_mask"]
        return np.random.choice(np.flatnonzero(possible_actions)), [], obs

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
