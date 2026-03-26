from dataclasses import dataclass
from typing import Union, Optional
from enum import Enum

class CardType(Enum):
    RULE = 1
    KEEPER = 2
    GOAL = 3
    ACTION = 4

class CardZone(Enum):
    RULES = 1,
    KEEPERS = 2,
    GOALS = 3,
    DISCARD_PILE = 4,
    HAND = 5

class ExtendedCardZone(Enum):
    ENEMY_KEEPERS = 1,
    OWN_KEEPERS = 2

AnyCardZone = Union[CardZone, ExtendedCardZone]

class GamePhaseType(Enum):
    PLAY_CARD_FOR_TURN = 1,
    POST_PLAY_CARD_FOR_TURN = 4,
    DISCARD_CARD_FROM_HAND = 2,
    DISCARD_KEEPER = 3,
    GAME_START = 5,
    TURN_END = 6,
    DISCARD_RULE_IN_PLAY = 7,

    def is_actionless(self):
        return self in [GamePhaseType.GAME_START, GamePhaseType.TURN_END, GamePhaseType.POST_PLAY_CARD_FOR_TURN]

@dataclass
class GamePhase:
    type: GamePhaseType
    acting_player: int
    decisions_left: Optional[int] = None

# Note: probably a good idea to make some of the other fields in this class optional
@dataclass
class GameState:
    """
    Note that the order of the lists is important.
    """
    hands: list[list[str]]
    keepers: list[list[str]]
    goals: list[str]
    discard_pile: list[str]
    draw_pile: list[str]
    rules: list[str]
    starting_player: Optional[int] = None

class DecisionEncodingType(Enum):
    PLAY = 0
    PLACE_PLAYER_HAND = 1
    PLACE_OPPONENT_HAND = 2
    PLACE_PLAYER_KEEPERS = 3
    PLACE_OPPONENT_KEEPERS = 4
    PLACE_DISCARD_PILE = 5
    PLACE_DRAW_PILE = 6
    PLACE_IN_PLAY = 7
    REMAIN_PLAYER_HAND = 8
    REMAIN_OPPONENT_HAND = 9
    REMAIN_PLAYER_KEEPERS = 10
    REMAIN_OPPONENT_KEEPERS = 11
    REMAIN_DISCARD_PILE = 12
    REMAIN_DRAW_PILE = 13
    REMAIN_IN_PLAY = 14