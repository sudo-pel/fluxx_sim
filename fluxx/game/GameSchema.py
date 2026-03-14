import abc

from fluxx.game.Card import Card, Rule, Goal

from fluxx.game.Player import Player
from Agents.Agent import Agent

class GameSchema(metaclass=abc.ABCMeta):
    def __init__(self, player_agents: list[Agent], deck: list[Card]):
        self.player_count: int = len(player_agents)
        self.players: list[Player] = []
        self.agents: list[Agent] = player_agents
        self.rules: list[Rule] = []
        self.player_turn: int = 0
        self.turn_count: int = 0
        self.goals: list[Goal] = []
        self.discard_pile: list[Card] = []
        self.draw_pile: list[Card] = []
        self.force_turn_over: bool = False
        self.winner = None
        self.deck = deck # for now, not the same as draw_pil e
        self.extra_turn = False
        self.played_free_actions = set()

        for i, agent in enumerate(self.agents):
            agent.set_player_number(i)
            self.players.append(Player(i))

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


