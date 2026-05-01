from typing import Any

import numpy.typing as npt
from pettingzoo.utils.env import ObsType

from src.game.Game import Game

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
cards have no runtime state in Fluxx so this is actually pretty reasonable

- Hand, Discard pile, Goals, Rules, Keepers for each player:all x sized vectors where x is the card pool size
- Always encode the observing player's keepers first. The rest can be in any order

- Action space: play a card, discard a card, discard a keeper (for now). Hot encode the card in question
    (( Augmenting the action space for action card effects ))
    - The action space is currently overcomplicated. The agent already knows what it is doing due to the presence of decision contexts, so there is no need to have copies ..
    .. of the hot encoding for aforementioned actions. The game can deduce what exactly to do based on the current GamePhase.
    

- Intermediate reward: how close you are to winning the game (iterate over goals and take a %) minus how close opponents are to winning the game
- Potentially a bonus for having a lot of cards in hand? Unsure

(( Expanding the state encoding to include decision contexts ))
Every card selection moves one card somewhere and "leaves" or "places" other cards elsewhere. The possible locations for being PLACED are:
- "being played"
- player hand
- opponent hand
- player keepers [* a keeper moving to this zone is not necessarily the same as playing it, e.g "steal a keeper"]
- opponent keepers
- discard pile
- draw pile

the possible locations for REMAINING are:
- player hand
- opponent hand
- player keepers
- opponent keepers
- discard pile
- draw pile

Also important to include is the number of "decisions" left. This is encoded as metadata in GamePhase, but is not used at all by the simulator.

Encoding is in this order:
[
PLACE
0: being played
1: player hand
2: opponent hand
3: player keepers
4: opponent keepers
5: discard pile
6: draw pile
7: in play [* goals and rules]
REMAIN
8: player hand
9: opponent hand
10: player keepers
11: opponent keepers
12: discard pile
13: draw pile
14: in play [* goals and rules]
15: DECISIONS_LEFT
]



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
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


def env(**kwargs):
    raw_env = FluxxEnv(**kwargs)
    raw_env = wrappers.AssertOutOfBoundsWrapper(raw_env)
    raw_env = wrappers.OrderEnforcingWrapper(raw_env)
    return raw_env

class FluxxEnv(AECEnv):

    metadata = {"render_modes": ["human"], "name": "card_game_v0"}

    def __init__(self, game: Game, num_players: int = 2, render_mode=None, seed: np.random.SeedSequence = None):
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

        # seed unused as of yet but added for completeness

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.terminations        = {agent: False for agent in self.agents}
        self.truncations         = {agent: False for agent in self.agents}
        self.rewards             = {agent: 0.0   for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0   for agent in self.agents}
        self.infos               = {agent: {}    for agent in self.agents}

        self.game.reset()

        # get player to move first
        acting_player = self.game.check_current_phase().acting_player
        self.agent_selection = f"player_{acting_player}"

        # Verify consistency
        phase = self.game.check_current_phase()
        assert self.game.player_turn == phase.acting_player, (
            f"Reset inconsistency: player_turn={self.game.player_turn}, "
            f"phase.acting_player={phase.acting_player}, "
            f"phase.type={phase.type.name}, "
            f"stack: {[(p.type.name, p.acting_player) for p in self.game.stack]}"
        )

        # return observation for the first player to act
        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self.game.step(action)

        # TODO: Intermediate rewards?
        # ...
        # Check termination
        if self.game.winner is not None:
            for agent in self.agents:
                self.terminations[agent] = True
                self.rewards[agent] = -1.0
            winner = self.determine_winner()
            self.rewards[winner] = 1.0

        # Accumulate rewards into _cumulative_rewards
        self._accumulate_rewards()

        # Advance to next agent (just check whats on top of the game stack)
        game_state = self.game.check_current_phase()
        self.agent_selection = f"player_{game_state.acting_player}"

    def observe(self, agent):
        return self.game.get_game_state()

    def decode_action(self, action_index: int) -> str:
        if action_index == self.card_vector_length:
            return "no_free_action"
        return self.index_to_card[action_index]

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

    def determine_winner(self):
        if self.game.winner == 0:
            return "player_0"
        elif self.game.winner == 1:
            return "player_1"
        else:
            raise Exception("determine_winner called with invalid winner")
