# begin with implementing a two-player game.

# card data is stored as an integer ID, which maps to relevant information
from enum import Enum
import random
from Card import CardType, Card, Rule, Goal, Keeper, Action
from Player import Player
from typing import Optional
from Agents.Agent import Agent

class Game:
    def __init__(self, player_agents: list[Agent]):
        self.player_count: int = len(player_agents)
        self.players: list[Player] = [Player(i) for i in range(self.player_count)]
        self.agents: list[Agent] = player_agents
        self.rules: list[Rule] = []
        self.turn: int = 0
        self.goal: Optional[Goal] = None
        self.discard_pile: list[Card] = []
        self.draw_pile: list[Card] = []
        self.winner = -1

    def get_draw_rules(self):
        """Calculate, based on cards in play, how many cards should be drawn by a player."""
        draw = 0
        for rule in self.rules:
            draw += rule.draw
        return draw

    def get_play_rules(self):
        """Calculate, based on cards in play, how many cards should be played by a player."""
        play = 0
        for rule in self.rules:
            play += rule.play
        return play

    def can_play_card(self, player: Player, i: int) -> bool:
        """
        Check whether a player can play a given card in their hand.

        Does not include input validation (checking whether 'i' is in range of hand array)
        """
        if player.cards_played >= self.get_play_rules():
            return False

        card_to_play = player.hand[i]

        return True

    def play_rule(self, player: Player, card_played: Rule):
        # Discard contradictory rules
        contradictory_rules = []
        possible_contradictions = ["draw", "play", "keeper_limit", "hand_limit"]
        for option in possible_contradictions:
            if card_played[option] > 0:
                contradictory_rules.append(option)

        new_rules = []
        for rule in self.rules:
            contradictory = False
            for option in contradictory_rules:
                if rule[option] > 0:
                    contradictory = True
                    break

            if not contradictory:
                new_rules.append(rule)

        self.rules = new_rules

        # Add new rule
        self.rules.append(card_played)

        # TODO: Rule immediate effects

        # TODO: Rule special effects

    def check_for_winners(self):
        """Check whether any players have won the game. If so, change 'winner' member variable"""
        if not self.goal:
            return

        goal_keepers = set(self.goal.required_keepers)

        for i, player in enumerate(self.players):
            keeper_names = [card.name for card in player.hand if card.card_type == CardType.KEEPER]
            if goal_keepers <= set(keeper_names):
                self.winner = i

        # TODO: special goal cards (cards in hand, etc)

    def play_goal(self, player: Player, goal: Goal):
        """Play a goal card. Does not perform validation."""
        self.goal = goal

        # Check whether any player has won the game
        self.check_for_winners()

    def play_keeper(self, player: Player, keeper: Keeper):
        """Play a keeper card. Does not perform validation."""
        player.keepers.append(keeper)

        self.check_for_winners()

    def play_action(self, player: Player, action: Action):
        pass

    def play_card(self, player: Player, i: int):
        """Play a card. Does not include validation."""
        card_to_play = player.hand[i]

        del player.hand[i]
        player.cards_played += 1

        if card_to_play.card_type == CardType.ACTION:
            self.play_action(player, card_to_play)
        elif card_to_play.card_type == CardType.RULE:
            self.play_rule(player, card_to_play)
        elif card_to_play.card_type == CardType.GOAL:
            self.play_goal(player, card_to_play)
        elif card_to_play.card_type == CardType.KEEPER:
            self.play_keeper(player, card_to_play)
        else:
            raise Exception("Invalid card type")

    def draw(self, player: Player):
        """Draw a card from the deck and add it to the hand of player 'player'."""
        card_drawn = self.draw_pile.pop()

        player.hand.append(card_drawn)

        if not self.draw_pile:
            random.shuffle(self.discard_pile)
            self.draw_pile = self.discard_pile
            self.discard_pile = []

    def start_of_turn(self):
        """Apply start of turn effects, including drawing."""
        turn_player = self.players[self.turn]

        # start-of-turn rule effects
        # draw
        draw_amount = self.get_draw_rules()
        # tba: special case re "play all but 1"

        # TODO: special start of turn effects

        for i in range(draw_amount):
            self.draw(turn_player)

    def end_of_turn(self):
        """Apply end of turn effects. Reset player turn stats like cards_played."""
        turn_player = self.players[self.turn]

        # end-of-turn rule effects

        self.turn = (self.turn + 1) % self.player_count


    def is_turn_over(self, ):
        """Check whether a players turn is over, based on cards played."""

    def run_game(self):
        while self.winner == -1:
            self.start_of_turn()

    # fluxx game flow
    # (1) start of turn: apply start-of-turn rule effects
    # (2) end-of-turn conditions reached?
    #   YES: end turn: go to (3)
    #   NO: prompt player game action
    #       apply player game action (see below)
    #       return to (2)
    # (3) apply end of turn effects
    # reset player turn stats (cards_drawn, cards_played, etc.)
    # increment self.turn
    # go to (1)

# test
new_game = Game()
