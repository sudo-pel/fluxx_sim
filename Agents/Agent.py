"""
Relevant agent choice points:
- play_free_action
- play_card
- discard_keeper
- discard_from_hand
- ... Many more for various action cards and free actions
"""

import abc

class Agent(metaclass=abc.ABCMeta):
    def __init__(self):
        self.player_number = -1

    def set_player_number(self, number: int):
        self.player_number = number

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'play_free_action') and
            callable(subclass.play_free_action) and
            hasattr(subclass, 'play_card') and
            callable(subclass.play_card) and
            hasattr(subclass, 'discard_keeper') and
            callable(subclass.discard_keeper) and
            hasattr(subclass, 'discard_from_hand') and
            callable(subclass.discard_from_hand)
            or
            NotImplemented
        )

    @abc.abstractmethod
    def play_card(self, game):
        raise NotImplementedError

    @abc.abstractmethod
    def discard_keeper(self, game):
        raise NotImplementedError

    @abc.abstractmethod
    def discard_from_hand(self, game):
        raise NotImplementedError

    @abc.abstractmethod
    def select_card(self, game, selection):
        raise NotImplementedError

    @abc.abstractmethod
    def select_player_besides_self(self, game):
        raise NotImplementedError

    @abc.abstractmethod
    def select_card_to_play(self, game, selection) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def select_player_from_set(self, game, player_numbers: list[int]):
        raise NotImplementedError

    @abc.abstractmethod
    def select_player_rotation_direction(self, game):
        raise NotImplementedError

    @abc.abstractmethod
    def play_free_action(self, game, available_free_actions: list[str]):
        raise NotImplementedError

    @abc.abstractmethod
    def choose_to_discard(self, game):
        raise NotImplementedError