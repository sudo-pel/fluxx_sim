import abc, random
from typing import Optional

from fluxx.game.Card import Card, Rule, Goal, make_card
from fluxx.game.FluxxEnums import GamePhaseType, GamePhase, GameState

from fluxx.game.Player import Player


class GameSchema(metaclass=abc.ABCMeta):
    def __init__(self, player_count: int, card_list: list[str], disable_game_messages: bool, force_game_state: Optional[GameState]):
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
        self.deck = card_list.copy()
        self.extra_turn = False
        self.played_free_actions = set()
        self.stack: list[GamePhase] = []
        self.disable_game_messages = disable_game_messages
        self.force_game_state = force_game_state

    def reset(self):
        random.shuffle(self.deck)
        self.draw_pile = [make_card(card_name) for card_name in self.deck]

        self.player_turn: int = random.randint(0, self.player_count - 1)
        self.turn_count: int = 0
        self.goals = []
        self.discard_pile = []
        self.rules = []
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

        if self.force_game_state is not None:
            self.draw_pile = []

            for i, hand in enumerate(self.force_game_state.hands):
                for card in hand:
                    self.players[i].hand.append(make_card(card))

            for i, keepers in enumerate(self.force_game_state.keepers):
                for card in keepers:
                    self.players[i].keepers.append(make_card(card))

            for goal in self.force_game_state.goals:
                self.goals.append(make_card(goal))

            for card in self.force_game_state.discard_pile:
                self.discard_pile.append(make_card(card))

            for card in self.force_game_state.draw_pile:
                self.draw_pile.append(make_card(card))

            for rule in self.force_game_state.rules:
                self.rules.append(make_card(rule))

            if self.force_game_state.starting_player is not None:
                self.player_turn = self.force_game_state.starting_player

    # TODO: Document/formalize game state. Consider making a class for game state
    # there is a class for game state although it is not currently used.
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


