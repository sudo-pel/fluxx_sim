from dataclasses import dataclass

from agents.Agent import Agent
from fluxx.game.FluxxEnums import GameConfig, GameState

"""

Implementation plan

- Function for checking whether a rule is a draw rule
- Function for checking whether a rule is a play rule


PLAY:
- Create a Gameplan object storing possible routes to victory. If any of the gameplans are "possible", promote that set of cards to the maximum, ensuring they come as a "set"
- Store a list of "asymmetric turn extender" cards
- Store a list of "symmetric turn extender" cards
- "Incomplete" gameplans should be discredited. 
- Calculate a list of goals that will cause your opponent to win the game and NEVER play them
- Similar logic for goals for which opponent has at least one keeper

DISCARD:
- Check what Gameplan the keeper/goal is a part of
    - If the gameplan has cards in the discard pile then increase the priority
    - If the gameplan has no cards in the discard pile then decrease the priority based on the number of gameplan cards missing (fewer missing = LOWER priority)
    
DISCARD RULE IN PLAY:
- Check whether rule is a "play", "draw", or "limit" rule and act accordingly

ADD CARD IN PLAY TO HAND:
- Map goals in play to gameplans and check whether the gameplan can be achieved this turn. If so, do NOT add the card in play to the hand
- ..If not, increase the priority inversely proportional to number of gameplan cards missing
- Also give priority to "limit" cards when you are over the limit

SHARE CARDS FROM LATENT SPACE:
- Map cards to gameplans and decrease priority based on number of gameplan cards missing
- Do the same for opponent, but with slightly smaller values
- Also prioritise asymmetric turn extender action cards, since they are good

STEAL CARD FROM OPPONENT:
- Map cards to gameplans and decrease priority based on number of gameplan cards missing
- Do the same for opponent, but with slightly smaller values

GIVE CARD TO OPPONENT:
- Inverse analysis to above

"""

@dataclass
class Gameplan:
    goal: str
    required_keepers: set[str]
    held_keepers: set[str]
    missing_keepers: set[str]
    have: int
    need: int

class HeuristicAgentMKI(Agent):
    def __init__(self, game_config: GameConfig, player_number: int):
        self.game_config = game_config
        self.player_number = player_number

    def act(self, state: GameState):
        pass
