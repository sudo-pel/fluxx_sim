# begin with implementing a two-player game.

# card data is stored as an integer ID, which maps to relevant information
from enum import Enum
import random
from typing import Optional

from Card import CardType, Card, Rule, Goal, Keeper, Action, RulesOptions

from Player import Player
from Agents.Agent import Agent
from Agents.PlayerControlled import PlayerControlledAgent

class Game:
    def __init__(self, player_agents: list[Agent], deck: list[Card]):
        self.player_count: int = len(player_agents)
        self.players: list[Player] = [Player(i) for i in range(self.player_count)]
        self.agents: list[Agent] = player_agents
        self.rules: list[Rule] = []
        self.player_turn: int = 0
        self.turn_count: int = 0
        self.goal: Optional[Goal] = None
        self.discard_pile: list[Card] = []
        self.draw_pile: list[Card] = []
        self.force_turn_over: bool = False
        self.winner = -1

    def get_draw_rules(self):
        """Calculate, based on cards in play, how many cards should be drawn by a player."""
        draw = 0
        for rule in self.rules:
            draw += rule.draw
        return draw

    def get_play_rules(self, player_number: int):
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
        turn_player = self.players[self.player_turn]

        # start-of-turn rule effects
        # draw
        draw_amount = self.get_draw_rules()
        # tba: special case re "play all but 1"

        # TODO: special start of turn effects

        for i in range(draw_amount):
            self.draw(turn_player)

    def end_of_turn(self):
        """Apply end of turn effects. Reset player turn stats like cards_played."""
        turn_player = self.players[self.player_turn]

        # end-of-turn rule effects

        self.player_turn = (self.player_turn + 1) % self.player_count
        self.turn_count += 1


    def player_turn_over(self) -> bool:
        """Check whether a players turn is over, based on cards played."""
        turn_player = self.players[self.player_turn]

        plays = self.get_play_rules(self.player_turn)

        if plays >= turn_player.cards_played:
            return True

        return self.force_turn_over

    # TODO: Document/formalize game state. Consider making a class for game state
    def get_game_state(self):
        return {
            "player_count": self.player_count,
            "players": self.players,
            "goal": self.goal,
            "rules": self.rules
        }

    def run_game(self):
        """Run the game."""
        while self.winner == -1:
            turn_agent = self.agents[self.player_turn]
            turn_player = self.players[self.player_turn]

            self.start_of_turn()

            """
            NOT IMPLEMENTED
            
            free_action = turn_agent.play_free_action()
            while free_action:
                # play free action
                free_action = turn_agent.play_free_action()
            """

            if self.player_turn_over():
                self.end_of_turn()
                continue

            played_card = turn_agent.play_card(self.get_game_state())

            if not self.can_play_card(turn_player, self.player_turn):
                raise Exception("Illegal action selected!")

            self.play_card(turn_player, played_card)



# test
test_deck = [
    Rule("hand_limit_2", RulesOptions(hand_limit=2)),
    Rule("hand_limit_1", RulesOptions(hand_limit=1)),
    Rule("hand_limit_0", RulesOptions(hand_limit=0)),
    Rule("keeper_limit_4", RulesOptions(keeper_limit=4)),
    Rule("keeper_limit_3", RulesOptions(keeper_limit=3)),
    Rule("keeper_limit_2", RulesOptions(keeper_limit=2)),
    Rule("draw_5", RulesOptions(draw=5)),
    Rule("draw_4", RulesOptions(draw=4)),
    Rule("draw_2", RulesOptions(draw=2)),
    Rule("play_2", RulesOptions(play=2)),
    Rule("draw_3", RulesOptions(draw=3)),
    Rule("play_4", RulesOptions(play=4)),
    Rule("play_3", RulesOptions(play=3)),

    Keeper("the_sun"),
    Keeper("the_party"),
    Keeper("music"),
    Keeper("dreams"),
    Keeper("love"),
    Keeper("peace"),
    Keeper("sleep"),
    Keeper("the_brain"),
    Keeper("bread"),
    Keeper("chocolate"),
    Keeper("cookies"),
    Keeper("milk"),
    Keeper("time"),
    Keeper("money"),
    Keeper("the_eye"),
    Keeper("the_moon"),
    Keeper("the_rocket"),
    Keeper("the_toaster"),
    Keeper("television"),

    Goal("the_appliances", ["the_toaster", "television"]),
    Goal("baked_goods", ["bread", "cookies"]),
    Goal("bed_time", ["sleep", "time"]),
    Goal("bread_and_chocolate", ["bread", "chocolate"]),
    Goal("cant_buy_me_love", ["money", "love"]),
    Goal("chocolate_cookies", ["chocolate", "cookies"]),
    Goal("chocolate_milk", ["chocolate", "milk"]),
    Goal("day_dreams", ["the_sun", "dreams"]),
    Goal("dreamland", ["sleep", "dreams"]),
    Goal("the_eye_of_the_beholder", ["the_eye", "love"]),
    Goal("great_theme_song", ["music", "television"]),
    Goal("hearts_and_minds", ["love", "the_brain"]),
    Goal("hippyism", ["peace", "love"]),
    Goal("lullaby", ["sleep", "music"]),
    Goal("milk_and_cookies", ["milk", "cookies"]),
    Goal("the_minds_eye", ["the_brain", "the_eye"]),
    Goal("night_and_day", ["the_sun", "the_moon"]),
    Goal("party_time", ["the_party", "time"]),
    Goal("rocket_science", ["the_rocket", "the_brain"]),
    Goal("rocket_to_the_moon", ["the_rocket", "the_moon"]),
    Goal("squishy_chocolate", ["chocolate", "the_sun"]),
    Goal("time_is_money", ["time", "money"]),
    Goal("toast", ["bread", "the_toasted"]),
    Goal("turn_it_up", ["music", "the_party"]),
    Goal("winning_the_lottery", ["dreams", "money"]),
    Goal("world_peace", ["dreams", "peace"])
]
new_game = Game([PlayerControlledAgent(), PlayerControlledAgent()], test_deck)
new_game.run_game()