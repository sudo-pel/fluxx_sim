# begin with implementing a two-player game.
import random
from typing import Optional
from collections import Counter

from fluxx.game.Card import Card, Rule, Goal, Keeper, Action, RulesOptions
from fluxx.game.FluxxEnums import CardType

from fluxx.game.Player import Player
from Agents.Agent import Agent
from fluxx.game import game_messages
from fluxx.game.Cards import action_cards
from fluxx.game.Cards.free_actions import can_use_free_action, activate_free_action
from fluxx.game.GameSchema import GameSchema
from fluxx.game.utils import card_effect_utils


class Game(GameSchema):
    def __init__(self, player_agents: list[Agent], deck: list[Card]):
        GameSchema.__init__(self, player_agents, deck)

    def reset(self):
        random.shuffle(self.deck)
        self.draw_pile = self.deck

        self.player_turn: int = 0
        self.turn_count: int = 0
        self.goals = []
        self.discard_pile = []
        self.draw_pile = []
        self.force_turn_over = False
        self.winner = None
        self.extra_turn = False
        self.played_free_actions = set()

        for player in self.players:
            player.hand = []
            player.keepers = []
            player.cards_drawn = 0
            player.cards_played = 0

    def run_game(self):
        """Run the game."""

        random.shuffle(self.deck)
        self.draw_pile = self.deck

        while self.winner is None:
            turn_agent = self.agents[self.player_turn]
            turn_player = self.players[self.player_turn]

            game_messages.turn_start(f"PLAYER {self.player_turn} TURN")
            self.start_of_turn()

            while self.winner is None:
                if self.player_turn_over():
                    self.end_of_turn()
                    break

                available_free_actions = self.get_available_free_actions()
                played_free_action = -1
                while len(available_free_actions) > 0 and played_free_action is not None and not self.force_turn_over:
                    played_free_action = turn_agent.play_free_action(self, available_free_actions)
                    if played_free_action is not None:
                        self.play_free_action(available_free_actions[played_free_action])
                        available_free_actions = self.get_available_free_actions()

                # Playing a free action can insta-end your turn
                if self.player_turn_over():
                    self.end_of_turn()
                    break

                self.draw_for_turn()

                played_card = None

                if self.rule_in_play("first_play_random"):
                    if self.get_play_rules(self.player_turn) > 1 and turn_player.cards_played == 0:
                        played_card = random.randint(0, len(turn_player.hand)-1)
                        game_messages.special_effect(f"<<< 'First Play Random': PLAYED {turn_player.hand[played_card].name}! >>")

                if played_card is None:
                    played_card = turn_agent.play_card(self.get_game_state())

                if not self.can_play_card(self.player_turn, played_card):
                    raise Exception("Illegal action selected!")

                self.play_card(self.player_turn, played_card)

        game_messages.game_over(f"GAME OVER: WINNER IS PLAYER {self.winner}")

    def get_draw_rules(self, player_number):
        """Calculate, based on cards in play, how many cards should be drawn by a player."""
        player = self.players[player_number]

        draw = 0
        for rule in self.rules:
            if rule.draw:
                draw += (rule.draw + self.inflation())

        if draw == 0:
            draw = 1 + self.inflation() # basic rule

        if self.rule_in_play("party_bonus") and self.keeper_in_play("the_party"):
            draw += (1 + self.inflation())

        if self.rule_in_play("poor_bonus"):
            keeper_count = len(player.keepers)
            keeper_counts = [len(p.keepers) for p in self.players]
            if keeper_count == min(keeper_counts) and Counter(keeper_counts)[keeper_count] == 1:
                draw += (1 + self.inflation())

        return draw

    def get_play_rules(self, player_number: int):
        """Calculate, based on cards in play, how many cards should be played by a player."""
        player = self.players[player_number]

        play = 0
        for rule in self.rules:
            if rule.play:
                play += (rule.play + self.inflation())

        if play == 0:
            play = 1 + self.inflation() # basic rule

        if self.rule_in_play("play_all"):
            play = player.cards_played + 1

        if self.rule_in_play("play_all_but_1") and len(player.hand) > 1 + (self.inflation()):
            play = player.cards_played + 1

        if self.rule_in_play("party_bonus") and self.keeper_in_play("the_party"):
            play += 1

        if self.rule_in_play("rich_bonus"):
            keeper_count = len(player.keepers)
            keeper_counts = [len(p.keepers) for p in self.players]
            if keeper_count == max(keeper_counts) and Counter(keeper_counts)[keeper_count] == 1:
                play += (1 + self.inflation())

        return play

    def get_keeper_limit(self, player_number: Optional[int]=None) -> Optional[int]:
        """Calculate how many keepers may be in play."""
        limit = None
        for rule in self.rules:
            if rule.keeper_limit is not None:
                limit = rule.keeper_limit + self.inflation()

        return limit

    def get_hand_limit(self, player_number: Optional[int]=None) -> Optional[int]:
        """Calculate how many cards may be held in hand."""
        limit = None
        for rule in self.rules:
            if rule.hand_limit is not None:
                limit = rule.hand_limit + self.inflation()

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

        self.check_for_winners()

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

    def play_rule(self, player_number: int, card_played: Rule):
        player = self.players[player_number]

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
        if not self.goals:
            return

        for current_goal in self.goals:
            # TODO: special goal cards (cards in hand, etc)
            if current_goal.name == "5_keepers":
                keeper_counts = [len(p.keepers) for p in self.players]
                for i, player in enumerate(self.players):
                    keeper_count = len(player.keepers)
                    if keeper_count >= 5 + self.inflation():
                        if keeper_count == max(keeper_counts) and Counter(keeper_counts)[keeper_count] == 1:
                            self.winner = i
                            return
                continue
            elif current_goal.name == "10_cards_in_hand":
                hand_counts = [len(p.hand) for p in self.players]
                for i, player in enumerate(self.players):
                    hand_count = len(player.hand)
                    if hand_count >= 10 + self.inflation():
                        if hand_count == max(hand_counts) and Counter(hand_counts)[hand_count] == 1:
                            self.winner = i
                            return
                continue

            goal_keepers = set(current_goal.required_keepers)

            for i, player in enumerate(self.players):
                keeper_names = set([card.name for card in player.keepers if card.card_type == CardType.KEEPER])
                if goal_keepers <= keeper_names:

                    win_cancelled = False
                    for goal_keeper in current_goal.disallowed_keepers:
                        if self.keeper_in_play(goal_keeper):
                            win_cancelled = True

                    optional_keepers = 0
                    for options in current_goal.optional_keepers:
                        for keeper in options:
                            if keeper in keeper_names:
                                optional_keepers += 1
                                break

                    if optional_keepers < len(current_goal.optional_keepers):
                        win_cancelled = True

                    if not win_cancelled:
                        self.winner = i
                        return

    def play_goal(self, player_number: int, goal: Goal):
        """Play a goal card. Does not perform validation."""
        player = self.players[player_number]

        if self.rule_in_play("double_agenda"):
            if len(self.goals) == 2:
                goal_location = card_effect_utils.select_card(self, player_number, ["goals"])
                card_effect_utils.trash_selected_card(self, player_number, goal_location, True)
        else:
            if len(self.goals) == 1:
                self.discard_pile.append(self.goals[0])
                del self.goals[0]

        self.goals.append(goal)

    def play_keeper(self, player_number: int, keeper: Keeper):
        """Play a keeper card. Does not perform validation."""
        player = self.players[player_number]

        player.keepers.append(keeper)

    def play_action(self, player_number: int, action: Action):
        action_cards.activate_action(action.name, self, player_number)
        self.discard_pile.append(action)

    def activate_card(self, player_number: int, card_to_play):
        """Activate a card. Is the result of 'playing a card', but is not the same thing as it"""
        if card_to_play.card_type == CardType.ACTION:
            self.play_action(player_number, card_to_play)
        elif card_to_play.card_type == CardType.RULE:
            self.play_rule(player_number, card_to_play)
        elif card_to_play.card_type == CardType.GOAL:
            self.play_goal(player_number, card_to_play)
        elif card_to_play.card_type == CardType.KEEPER:
            self.play_keeper(player_number, card_to_play)
        else:
            raise Exception("Invalid card type")

    def play_card(self, player_number: int, i: int):
        player = self.players[player_number]

        """Play a card. Does not include validation. Discards card from player hand but does NOT add it to the discard pile"""
        card_to_play = player.hand[i]

        del player.hand[i]
        player.cards_played += 1

        self.activate_card(player_number, card_to_play)

        self.check_for_winners()

    def get_card_from_draw_pile(self) -> Card:
        """Get a card from the draw pile. Not the same as drawing a card."""
        card_drawn = self.draw_pile.pop()
        game_messages.drawn_card(f"[[ DRAWN '{card_drawn.name}' ]]")

        if not self.draw_pile:
            random.shuffle(self.discard_pile)
            self.draw_pile = self.discard_pile
            self.discard_pile = []

        return card_drawn

    def draw(self, player: Player):
        """Draw a card from the deck and add it to the hand of player 'player'. Does NOT increment player.cards_drawn"""
        card_drawn = self.get_card_from_draw_pile()

        player.hand.append(card_drawn)
        self.check_for_winners()

    def draw_for_turn(self):
        """Draw cards for the turn player."""
        turn_player = self.players[self.player_turn]

        # start-of-turn rule effects
        # draw
        draw_amount = self.get_draw_rules(self.player_turn)
        previous_cards_in_hand = len(turn_player.hand)

        while turn_player.cards_drawn < draw_amount:
            self.draw(turn_player)
            turn_player.cards_drawn += 1

        if self.rule_in_play("play_all_but_1"):
            if previous_cards_in_hand == 0 and draw_amount == 1 + (self.inflation()):
                self.draw(turn_player)
                turn_player.cards_drawn += 1

    def start_of_turn(self):
        """Apply start of turn effects, including drawing."""
        turn_player = self.players[self.player_turn]

        if self.rule_in_play("no_hand_bonus"):
            if len(turn_player.hand) == 0:
                for i in range(3 + self.inflation()):
                    self.draw(turn_player)

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
        self.played_free_actions = set()

        if not self.extra_turn:
            self.player_turn = (self.player_turn + 1) % self.player_count
        else:
            self.extra_turn = False

        self.turn_count += 1


    def player_turn_over(self) -> bool:
        """Check whether a players turn is over, based on cards played."""
        turn_player = self.players[self.player_turn]

        # turn over if out of plays
        plays = self.get_play_rules(self.player_turn)
        if plays <= turn_player.cards_played:
            return True

        # turn over if out of cards in hand + no free actions
        if len(turn_player.hand) == 0:
            return True

        return self.force_turn_over

    def get_available_free_actions(self):
        available_free_actions = []

        for rule in self.rules:
            if rule.free_action and can_use_free_action(rule.name, self, self.player_turn):
                available_free_actions.append(rule.name)

        return available_free_actions

    def play_free_action(self, free_action_name):
        activate_free_action(free_action_name, self, self.player_turn)
        self.played_free_actions.add(free_action_name)

        self.check_for_winners()

    def rule_in_play(self, rule_name) -> bool:
        for rule in self.rules:
            if rule.name == rule_name:
                return True

        return False

    def keeper_in_play(self, keeper_name) -> bool:
        for player in self.players:
            for keeper in player.keepers:
                if keeper.name == keeper_name:
                    return True

        return False

    def inflation(self):
        if self.rule_in_play("inflation"):
            #game_messages.notification("(Inflation bonus)")
            return 1
        else:
            return 0



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
    Goal("world_peace", ["dreams", "peace"]),

    Action("use_what_you_take"),
    Action("zap_a_card"),
    Action("trash_a_new_rule"),
    Action("trash_a_keeper"),
    Action("trade_hands"),
    Action("todays_special"),
    Action("draw_2_and_use_em"),
    Action("draw_3_play_2_of_them"),
    Action("steal_a_keeper"),
    Action("share_the_wealth"),
    Action("rules_reset"),
    Action("rock_paper_scissors_showdown"),
    Action("random_tax"),
    Action("no_limits"),
    Action("lets_simplify"),
    Action("lets_do_that_again"),
    Action("jackpot"),
    Action("exchange_keepers"),
    Action("empty_the_trash"),
    Action("discard_and_draw"),
    Action("take_another_turn"),
    Action("rotate_hands"),
    Action("everybody_gets_1"),

    Rule("mystery_play", RulesOptions(free_action=True)),
    Rule("swap_plays_for_draws", RulesOptions(free_action=True)),
    Rule("get_on_with_it", RulesOptions(free_action=True)),
    Rule("recycling", RulesOptions(free_action=True)),
    Rule("goal_mill", RulesOptions(free_action=True)),
    Rule("play_all", RulesOptions(play=-1)),
    Rule("play_all_but_1", RulesOptions(play=-1)),
    Rule("no_hand_bonus", RulesOptions()),
    Rule("party_bonus", RulesOptions()),
    Rule("poor_bonus", RulesOptions()),
    Rule("rich_bonus", RulesOptions()),
    Rule("double_agenda", RulesOptions()),
    Rule("first_play_random", RulesOptions()),
    Rule("inflation", RulesOptions()),

    Goal("5_keepers", []),
    Goal("10_cards_in_hand", []),
    Goal("the_brain_no_tv", ["the_brain"], ["television"]),
    Goal("party_snacks", ["the_party"], [], [["milk", "cookies", "chocolate", "bread"]]),
]