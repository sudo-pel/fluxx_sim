# begin with implementing a two-player game.
import random
from typing import Optional

from Card import CardType, Card, Rule, Goal, Keeper, Action, RulesOptions

from Player import Player
from Agents.Agent import Agent
from Agents.PlayerControlled import PlayerControlledAgent

class Game:
    def __init__(self, player_agents: list[Agent], deck: list[Card]):
        self.player_count: int = len(player_agents)
        self.players: list[Player] = []
        self.agents: list[Agent] = player_agents
        self.rules: list[Rule] = []
        self.player_turn: int = 0
        self.turn_count: int = 0
        self.goal: Optional[Goal] = None
        self.discard_pile: list[Card] = []
        self.draw_pile: list[Card] = []
        self.force_turn_over: bool = False
        self.winner = -1
        self.deck = deck # for now, not the same as draw_pile

        for i, agent in enumerate(self.agents):
            agent.set_player_number(i)
            self.players.append(Player(i))

    def get_draw_rules(self):
        """Calculate, based on cards in play, how many cards should be drawn by a player."""
        draw = 0
        for rule in self.rules:
            if rule.draw:
                draw += rule.draw

        if draw == 0:
            draw = 1 # basic rule

        return draw

    def get_play_rules(self, player_number: int):
        """Calculate, based on cards in play, how many cards should be played by a player."""
        play = 0
        for rule in self.rules:
            if rule.play:
                play += rule.play

        if play == 0:
            play = 1 # basic rule

        return play

    def get_keeper_limit(self, player_number: Optional[int]=None) -> Optional[int]:
        """Calculate how many keepers may be in play."""
        limit = None
        for rule in self.rules:
            if rule.keeper_limit is not None:
                limit = rule.keeper_limit

        return limit

    def get_hand_limit(self, player_number: Optional[int]=None) -> Optional[int]:
        """Calculate how many cards may be held in hand."""
        limit = None
        for rule in self.rules:
            if rule.hand_limit is not None:
                limit = rule.hand_limit

        return limit

    def can_play_card(self, player_number: int, i: int) -> bool:
        """
        Check whether a player can play a given card in their hand.

        Does not include input validation (checking whether 'i' is in range of hand array)
        """
        player = self.players[player_number]

        if player.cards_played >= self.get_play_rules(player_number):
            return False

        card_to_play = player.hand[i]

        return True

    def discard_card(self, player: Player, card_to_discard: int):
        self.discard_pile.append(player.hand[card_to_discard])
        del player.hand[card_to_discard]

    def discard_keeper(self, player: Player, keeper_to_discard: int):
        self.discard_pile.append(player.keepers[keeper_to_discard])
        del player.keepers[keeper_to_discard]

    def limit_check_player(self, player_number: int):
        player = self.players[player_number]

        keeper_limit = self.get_keeper_limit()
        hand_limit = self.get_hand_limit()

        while keeper_limit is not None and len(player.keepers) > keeper_limit:
            keeper_to_discard = self.agents[player_number].discard_keeper(self.get_game_state())
            self.discard_keeper(player, keeper_to_discard)

        while hand_limit is not None and len(player.hand) > hand_limit:
            card_to_discard = self.agents[player_number].discard_from_hand(self.get_game_state())
            self.discard_card(player, card_to_discard)

    def play_rule(self, player: Player, card_played: Rule):
        # Discard contradictory rules
        contradictory_rules = []
        possible_contradictions = ["draw", "play", "keeper_limit", "hand_limit"]
        for option in possible_contradictions:
            if card_played[option]:
                contradictory_rules.append(option)

        new_rules = []
        for rule in self.rules:
            contradictory = False
            for option in contradictory_rules:
                if rule[option] is not None:
                    contradictory = True
                    self.discard_pile.append(rule)
                    break

            if not contradictory:
                new_rules.append(rule)

        self.rules = new_rules

        # Add new rule
        self.rules.append(card_played)

        # Rule immediate effects (limits)
        for i, cur_player in enumerate(self.players):
            if cur_player == player:
                continue

            self.limit_check_player(i)

        # TODO: Rule special effects

    def check_for_winners(self):
        """Check whether any players have won the game. If so, change 'winner' member variable"""
        if not self.goal:
            return

        goal_keepers = set(self.goal.required_keepers)

        for i, player in enumerate(self.players):
            keeper_names = [card.name for card in player.keepers if card.card_type == CardType.KEEPER]
            if goal_keepers <= set(keeper_names):
                print(f"[[ THE GAME HAS BEEN WON BY PLAYER {i} !!! ]]")
                self.winner = i

        # TODO: special goal cards (cards in hand, etc)

    def play_goal(self, player: Player, goal: Goal):
        """Play a goal card. Does not perform validation."""
        if self.goal:
            self.discard_pile.append(self.goal)
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
        """Play a card. Does not include validation. Discards card from player hand but does NOT add it to the discard pile"""
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
        """Draw a card from the deck and add it to the hand of player 'player'. Does NOT increment player.cards_drawn"""
        card_drawn = self.draw_pile.pop()
        print(f"[[ DRAWN '{card_drawn.name}' ]]")

        player.hand.append(card_drawn)

        if not self.draw_pile:
            random.shuffle(self.discard_pile)
            self.draw_pile = self.discard_pile
            self.discard_pile = []

    def draw_for_turn(self):
        """Draw cards for the turn player."""
        turn_player = self.players[self.player_turn]

        # start-of-turn rule effects
        # draw
        draw_amount = self.get_draw_rules()
        print(f"draw amount: {draw_amount}")
        # tba: special case re "play all but 1"

        # TODO: special start of turn effects

        while turn_player.cards_drawn < draw_amount:
            self.draw(turn_player)
            turn_player.cards_drawn += 1
            print(f"drawn {turn_player.cards_drawn}/{draw_amount}")

    def start_of_turn(self):
        """Apply start of turn effects, including drawing."""
        self.draw_for_turn()

    def end_of_turn(self):
        """Apply end of turn effects. Reset player turn stats like cards_played."""
        turn_player = self.players[self.player_turn]

        # end-of-turn rule effects

        # end-of-turn limit checks
        self.limit_check_player(self.player_turn)

        # reset turn statistics
        turn_player.cards_drawn = 0
        turn_player.cards_played = 0
        self.force_turn_over = False

        self.player_turn = (self.player_turn + 1) % self.player_count
        self.turn_count += 1


    def player_turn_over(self) -> bool:
        """Check whether a players turn is over, based on cards played."""
        turn_player = self.players[self.player_turn]

        # turn over if out of plays
        plays = self.get_play_rules(self.player_turn)
        print(f"plays: {turn_player.cards_played}")
        if plays <= turn_player.cards_played:
            return True

        # turn over if out of cards in hand + no free actions
        if len(turn_player.hand) == 0: # TODO: free actions check
            return True

        return self.force_turn_over

    # TODO: Document/formalize game state. Consider making a class for game state
    def get_game_state(self):
        return {
            "player_count": self.player_count,
            "players": self.players,
            "goal": self.goal,
            "rules": self.rules,
            "player_turn": self.player_turn
        }

    def run_game(self):
        """Run the game."""
        random.shuffle(self.deck)
        self.draw_pile = self.deck

        while self.winner == -1:
            turn_agent = self.agents[self.player_turn]
            turn_player = self.players[self.player_turn]

            self.start_of_turn()

            while True:
                """
                NOT IMPLEMENTED

                free_action = turn_agent.play_free_action()
                while free_action:
                    # play free action
                    free_action = turn_agent.play_free_action()
                """

                if self.player_turn_over():
                    self.end_of_turn()
                    break

                self.draw_for_turn()

                played_card = turn_agent.play_card(self.get_game_state())

                if not self.can_play_card(self.player_turn, played_card):
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