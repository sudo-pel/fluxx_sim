import abc, random

from fluxx.game.Card import Card, Rule, Goal, make_card
from fluxx.game.FluxxEnums import GamePhaseType, GamePhase

from fluxx.game.Player import Player


class GameSchema(metaclass=abc.ABCMeta):
    def __init__(self, player_count: int, card_list: list[str], disable_game_messages: bool):
        self.card_list = card_list # static, not shuffled
        self.player_count: int = player_count
        self.players: list[Player] = [Player(i) for i in range(player_count)]
        self.rules: list[Rule] = []
        self.player_turn: int = 0
        self.turn_count: int = 0
        self.goals: list[Goal] = []
        self.discard_pile: list[Card] = []
        self.draw_pile: list[Card] = []
        self.force_turn_over: bool = False
        self.winner = None
        self.deck = card_list
        self.extra_turn = False
        self.played_free_actions = set()
        self.stack: list[GamePhase] = [GamePhase(GamePhaseType.GAME_START, -1)]
        self.disable_game_messages = disable_game_messages

    def reset(self):
        random.shuffle(self.deck)
        self.draw_pile = [make_card(card_name) for card_name in self.deck]

        self.player_turn: int = 0
        self.turn_count: int = 0
        self.goals = []
        self.discard_pile = []
        self.force_turn_over = False
        self.winner = None
        self.extra_turn = False
        self.played_free_actions = set()
        self.stack = [GamePhase(GamePhaseType.GAME_START, -1)]

        for player in self.players:
            player.hand = []
            player.keepers = []
            player.cards_drawn = 0
            player.cards_played = 0

    # TODO: Document/formalize game state. Consider making a class for game state
    def get_game_state(self):
        return {
            "player_count": self.player_count,
            "players": self.players,
            "goals": self.goals,
            "rules": self.rules,
            "player_turn": self.player_turn
        }

    @abc.abstractmethod
    def get_card_from_draw_pile(self):
        raise NotImplementedError

    @abc.abstractmethod
    def activate_card(self, user_number: int, selected_card: Card):
        raise NotImplementedError


