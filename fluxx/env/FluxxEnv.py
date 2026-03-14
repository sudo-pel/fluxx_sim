import numpy as np
import gymnasium as gym

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

class FluxxEnv(gym.Env):

    def __init__(self, game: Game):
        self.game = Game

