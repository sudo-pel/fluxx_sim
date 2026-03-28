from dataclasses import dataclass
from typing import Union, Optional
from enum import Enum

class CardType(Enum):
    RULE = 1
    KEEPER = 2
    GOAL = 3
    ACTION = 4

class Card:
    def __init__(self, name: str, card_type: CardType):
        self.name = name
        self.card_type = card_type

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
    PLAY_CARD_FROM_LATENT_SPACE = 8,
    ADD_CARD_IN_PLAY_TO_HAND = 9,
    SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND = 10,
    PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE = 11,
    DEFERRED_ADD_CARD_TO_DISCARD_PILE = 12,
    DISCARD_KEEPER_IN_PLAY = 13,
    PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT = 14,
    SELECT_KEEPER_TO_STEAL = 15,
    SELECT_PLAYER_KEEPER_FOR_EXCHANGE = 16,
    SELECT_OPPONENT_KEEPER_FOR_EXCHANGE = 17,
    ACTIVATE_FREE_ACTION = 18,
    DISCARD_OWN_KEEPER_IN_PLAY = 19,
    DEFERRED_DRAW_CARD = 20, # Exists when card draw needs to be deferred (e.g "recycling")
    DISCARD_VARIABLE_CARDS_FROM_HAND = 21,

    def is_actionless(self):
        return self in {GamePhaseType.GAME_START, GamePhaseType.TURN_END, GamePhaseType.POST_PLAY_CARD_FOR_TURN, GamePhaseType.DEFERRED_ADD_CARD_TO_DISCARD_PILE, GamePhaseType.DEFERRED_DRAW_CARD}

    def contains_latent_space(self):
        return self in {GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE, GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND, GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT}

class OnCompleteBehaviour(Enum):
    DRAW = 1

@dataclass
class GamePhase:
    type: GamePhaseType
    acting_player: int
    decisions_left: Optional[int] = None
    latent_space: Optional[list[Card]] = None
    card: Optional[Card] = None
    labelled_card: Optional[Card] = None # "labelled_card is not counted in card conservation unlike regular "card", which is considered to be in "latent_space"
    counter: Optional[int] = None
    card_types: Optional[set[CardType]] = None
    on_complete: Optional[OnCompleteBehaviour] = None

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
    PLAY_FOR_OPPONENT = 15