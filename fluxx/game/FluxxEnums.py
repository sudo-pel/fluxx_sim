from dataclasses import dataclass
from typing import Union
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

@dataclass
class GamePhase:
    type: GamePhaseType
    acting_player: int

class GameActionType(Enum):
    PLAY_CARD_FOR_TURN = 1,
    DISCARD_CARD_FROM_HAND = 2,
    DISCARD_KEEPER = 3,
    NULL_ACTION = 4, # Called only by environment. Messy solution, may not keep

@dataclass
class GameAction:
    type: GameActionType
    card_name: str