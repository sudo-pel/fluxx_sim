import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from typing import Optional

from fluxx.game.FluxxEnums import GamePhaseType, GameAction, GameActionType
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
- Always encode the observing player's keepers first. The rest can be in any order

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
        self.card_vector_length = len(game.card_list)
        self.card_to_index: dict[str, int] = {
            card: i for i, card in enumerate(game.card_list)
        }
        self.index_to_card = {
            i: card for i, card in enumerate(game.card_list)
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

        # return observation for the first player to act
        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self.game.step(action)

        # TODO: Update rewards
        # ...
        # TODO: Check termination
        if self.game.winner is not None:
            for agent in self.agents:
                self.terminations[agent] = True

        # TODO: Accumulate rewards into _cumulative_rewards
        # ...
        # TODO: Advance to next agent (just check whats on top of the game stack)
        game_state = self.game.check_current_phase()
        self.agent_selection = f"player_{game_state.acting_player}"

    def observe(self, agent):
        # TODO: return observation and action mask for this agent

        # ----
        # OBSERVATION
        # ----

        # Get all keepers in play
        keepers_in_play = self.game.get_all_keepers_by_name()
        keeper_vectors = [ self.populate_card_vector(keeper_list) for keeper_list in keepers_in_play ]
        agent_keeper_vector = keeper_vectors[self.get_player_number(agent)]
        other_keeper_vectors = keeper_vectors[:self.get_player_number(agent)] + keeper_vectors[self.get_player_number(agent)+1:]

        # Get cards in hand
        cards_in_hand = self.game.get_cards_in_hand_by_name(self.get_player_number(agent))
        cards_in_hand_vector = self.populate_card_vector(cards_in_hand)

        # Get goals in play
        goals_in_play = self.game.get_goals_in_play_by_name()
        goals_in_play_vector = self.populate_card_vector(goals_in_play)

        # Get rules in play
        rules_in_play = self.game.get_rules_in_play_by_name()
        rules_in_play_vector = self.populate_card_vector(rules_in_play)

        observation = np.concatenate((cards_in_hand_vector, agent_keeper_vector, *other_keeper_vectors, goals_in_play_vector, rules_in_play_vector))

        # ----
        # ACTION MASK
        # ----

        # Determine what actions are legal based on the game phase
        current_phase = self.game.check_current_phase()

        play_card_mask = np.zeros(self.card_vector_length, dtype=np.int8)
        discard_card_from_hand_mask = np.zeros(self.card_vector_length, dtype=np.int8)
        discard_keeper_mask = np.zeros(self.card_vector_length, dtype=np.int8)

        # TODO: Mask *in* legal plays (cards in hand, keepers owned) and then return the concatenation of all
        if current_phase.type == GamePhaseType.PLAY_CARD_FOR_TURN:
            play_card_mask = cards_in_hand_vector
        elif current_phase.type == GamePhaseType.DISCARD_CARD_FROM_HAND:
            discard_card_from_hand_mask = cards_in_hand_vector
        elif current_phase.type == GamePhaseType.DISCARD_KEEPER:
            discard_keeper_mask = agent_keeper_vector

        action_mask = np.concatenate((play_card_mask, discard_card_from_hand_mask, discard_keeper_mask))

        # Get action_mask
        return {
            "observation": observation,
            "action_mask": action_mask
        }

    def decode_action(self, action_index: int) -> GameAction:
        action_types = [GameActionType.PLAY_CARD_FOR_TURN, GameActionType.DISCARD_CARD_FROM_HAND, GameActionType.DISCARD_KEEPER]

        action_type = action_index // self.card_vector_length
        card_name = self.index_to_card[action_index % self.card_vector_length]

        return GameAction(action_types[action_type], card_name)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        pass

    def get_player_number(self, agent) -> int:
        """
        Convert agent identifier ('player_0') to player number (0)
        """
        return self.possible_agents.index(agent)

    def populate_card_vector(self, card_list: list[str]) -> npt.NDArray[np.int8]:
        vector = np.zeros(self.card_vector_length, dtype=np.int8)

        # TODO: vectorise this
        for card in card_list:
            vector[self.card_to_index[card]] = 1
        return vector