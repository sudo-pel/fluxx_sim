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