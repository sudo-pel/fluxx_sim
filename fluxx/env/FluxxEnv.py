import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from typing import Optional

from fluxx.game.Game import Game
"""

[[ Encoding game state ]]

Need to encode the following:
- Hand for observing player
- Keepers for each player
- Rules
- Goals

[[STARTINGWITHTHIS]]
--- approach one ---
Sounds reasonable to actually just each as a large vector, hot encoding if a given card is in that zone
Cards have no runtime state in Fluxx so this is actually pretty reasonable

- Hand, Discard pile, Goals, Rules, Keepers for each player:all x sized vectors where x is the card pool size

- Action space: play a card, discard a card, discard a keeper (for now). Hot encode the card in question

- Intermediate reward: how close you are to winning the game (iterate over goals and take a %) minus how close opponents are to winning the game
- Potentially a bonus for having a lot of cards in hand? Unsure

--- approach two ---
Ideally, RL agent is able to learn that playing cards with related goals is good "in general", for this a more general encoding is needed than one-hot encoding each state

Possible encoding for keepers:
- card ID (?)
- how many goals for this keeper are in the {DISCARD_PILE, HAND, IN_PLAY} (3 integers)

Possible encoding for goals:
- card ID
- card ID of required keepers (padded list)
- completion ratio (what proportion of the required goals are in each location) {DISCARD_PILE, HAND, IN_PLAY for user, IN_PLAY for each opponent)

Possible encoding for rules:
- card ID
- play, count modifiers (2 integers: -1 if no impact)
- mechanical embedding

Possible encoding for actions:
- card ID
- mechanical embedding

"""

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def env(**kwargs):
    raw_env = FluxxEnv(**kwargs)
    raw_env = wrappers.AssertOutOfBoundsWrapper(raw_env)
    raw_env = wrappers.OrderEnforcingWrapper(raw_env)
    return raw_env


class FluxxEnv(AECEnv):

    metadata = {"render_modes": ["human"], "name": "card_game_v0"}

    def __init__(self, game: Game, num_players: int = 2, render_mode=None):
        super().__init__()

        self.game = game
        self.card_vector = [0 for x in game.deck]
        self.card_map = {
            i: card_name for i, card_name in enumerate(game.deck)
        }

        self.num_players = num_players
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        observed_zone_count = 3 + num_players # hand (for observing agent), goals, rules, keepers (for each agent)
        observation_space_size = observed_zone_count * len(game.deck)

        possible_actions = 3 # play a card, discard a card, discard a keeper
        action_space_size = possible_actions * len(game.deck)

        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(observation_space_size,), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(action_space_size,), dtype=np.int8),
            })
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(action_space_size)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.terminations        = {agent: False for agent in self.agents}
        self.truncations         = {agent: False for agent in self.agents}
        self.rewards             = {agent: 0.0   for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0   for agent in self.agents}
        self.infos               = {agent: {}    for agent in self.agents}

        self.game.reset()

        # player 0 goes first
        self.agent_selection = "player_0"

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0

        # TODO: apply action, update game state, assign rewards, set terminations

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        # TODO: return observation and action mask for this agent
        return {
            "observation": np.zeros(10, dtype=np.float32),
            "action_mask": np.ones(52, dtype=np.int8),
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        pass