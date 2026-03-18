# begin with implementing a two-player game.
import math
import random
from typing import Optional
from collections import Counter

from fluxx.game.Card import Card, Rule, Goal, Keeper, Action, RulesOptions
from fluxx.game.FluxxEnums import CardType, GamePhase, GamePhaseType, GameAction, GameActionType

from fluxx.game.Player import Player
from fluxx.game import game_messages
from fluxx.game.Cards import action_cards
from fluxx.game.Cards.free_actions import can_use_free_action, activate_free_action
from fluxx.game.GameSchema import GameSchema
from fluxx.game.utils import card_effect_utils
from fluxx.game.utils.general_utils import index_of_card


class Game(GameSchema):
    def __init__(self, player_count: int, card_list: list[str]):
        GameSchema.__init__(self, player_count, card_list)

    def reset(self):
        super().reset()
        self.step(GameAction(GameActionType.NULL_ACTION, ""))

    def check_current_phase(self) -> GamePhase:
        if len(self.stack) == 0:
            return GamePhase(GamePhaseType.PLAY_CARD_FOR_TURN, self.player_turn)
        else:
            return self.stack[-1]

    def get_current_phase(self) -> GamePhase:
        """
        The same as check_current_phase, but pops from the phase stack
        """
        if len(self.stack) == 0:
            return GamePhase(GamePhaseType.PLAY_CARD_FOR_TURN, self.player_turn)
        else:
            return self.stack.pop()

    def is_action_valid(self, phase: GamePhase, action: GameAction) -> tuple[bool, Optional[str]]:
        """
        Given a GamePhase and a game state (implicit in the self-reference), validates a given GameAction.

        First, checks whether the game action corresponds to the given phase. Then performs validation specific to the given GameAction.
        """

        # Consider refactoring this into a dict with a list in the future
        if phase.type == GamePhaseType.PLAY_CARD_FOR_TURN and action.type != GameActionType.PLAY_CARD_FOR_TURN: return False, "Invalid action type"
        if phase.type == GamePhaseType.DISCARD_CARD_FROM_HAND and action.type != GameActionType.DISCARD_CARD_FROM_HAND: return False, "Invalid action type"
        if phase.type == GamePhaseType.DISCARD_KEEPER and action.type != GameActionType.DISCARD_KEEPER: return False, "Invalid action type"

        # Specific action validation
        if action.type == GameActionType.PLAY_CARD_FOR_TURN:
            # is it this player's turn?
            if self.player_turn != phase.acting_player: return False, "Not this player's turn"

            # is the card in this player's hard?
            if action.card_name not in self.get_cards_in_hand_by_name(phase.acting_player): return False, "Card to be played not in hand"

        elif action.type == GameActionType.DISCARD_CARD_FROM_HAND:
            # is the card in this player's hand?
            if action.card_name not in self.get_cards_in_hand_by_name(phase.acting_player): return False, "Card to be discarded not in hand"

        elif action.type == GameActionType.DISCARD_KEEPER:
            # does the player own this keeper?
            if action.card_name not in self.get_keepers_by_name(phase.acting_player): return False, "Keeper to be discarded not owned"

        return True, None

    # We note that the simulator will receive an integer and then decode it into something more complex for the game simulator to consume
    def step(self, action: GameAction):
        current_phase = self.get_current_phase()
        acting_player = current_phase.acting_player

        action_valid, error_message = self.is_action_valid(current_phase, action)
        if not action_valid:
            raise Exception(f"Invalid action: {error_message}")

        if action.type == GameActionType.PLAY_CARD_FOR_TURN:
            card_to_play = action.card_name
            self.play_card_from_hand(acting_player, card_to_play)
        elif action.type == GameActionType.DISCARD_CARD_FROM_HAND:
            card_to_discard = action.card_name
            self.discard_card(acting_player, card_to_discard)
        elif action.type == GameActionType.DISCARD_KEEPER:
            keeper_to_discard = action.card_name
            self.discard_keeper(acting_player, keeper_to_discard)

        # TODO: Implement "actionless phases": for example, after playing a card, we want to return to post_play_card_for_turn which has some end-of-phase logic. To do this, we will append to the stack (POST_PLAY_FOR_TURN, player_number) immediately after processing PLAY_CARD_FOR_TURN. The simulator must then remember to deal with this in the future, whether it is immediately after playing this card (in most cases) or a long time after some action processing, discarding of cards, etc.
        # caveat: GAME_START is an actionless phase, but it will be called at the start of the game via a step() function and therefore will not be at the top of the stack (it will have been popped into current_phase)
        if current_phase.type == GamePhaseType.GAME_START:
            # Start of game: begin the turn of the first player
            self.start_of_turn()
            self.stack.append(GamePhase(GamePhaseType.POST_PLAY_CARD_FOR_TURN, self.player_turn))
            self.stack.append(GamePhase(GamePhaseType.PLAY_CARD_FOR_TURN, self.player_turn))

        elif self.check_current_phase().type == GamePhaseType.POST_PLAY_CARD_FOR_TURN:
            self.get_current_phase()
            self.handle_turn_over()

        elif self.check_current_phase().type == GamePhaseType.TURN_END:
            self.get_current_phase()
            self.end_of_turn()
            self.start_of_turn()
            self.stack.append(GamePhase(GamePhaseType.POST_PLAY_CARD_FOR_TURN, self.player_turn))
            self.stack.append(GamePhase(GamePhaseType.PLAY_CARD_FOR_TURN, self.player_turn))

    def handle_turn_over(self):
        """
        Checks whether the turn player's turn is over, advancing the turn if so and adding another PLAY+POST_CARD_FOR_TURN to the stack otherwise
        """
        if self.is_turn_over():
            self.handle_end_of_turn()
        else:
            # Allow the player to play another card
            self.stack.append(GamePhase(GamePhaseType.POST_PLAY_CARD_FOR_TURN, self.player_turn))
            self.stack.append(GamePhase(GamePhaseType.PLAY_CARD_FOR_TURN, self.player_turn))

    def handle_end_of_turn(self):
        """
        Prepare end-of-turn logic by adding the necessary stack actions. Actual "ending of the turn" handled in game.step()/end_of_turn().
        """
        self.stack.append(GamePhase(GamePhaseType.TURN_END, self.player_turn))
        # game.limit_check_player() will add necessary discarding/other EoT stack events
        self.limit_check_player(self.player_turn)

    def end_of_turn(self):
        """
        Reset turn statistics
        """
        turn_player = self.players[self.player_turn]

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

    # DEPRECATED
    def run_game(self):
        """Run the game."""
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
                        free_action_name = available_free_actions[played_free_action]

                        self.play_free_action(free_action_name)
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

    def discard_card(self, player_number: int, card_name: str):
        """
        Discard a card from a player's hand. Does not check whether the card is actually in the player's hand.
        """
        player = self.players[player_number]

        card_index = index_of_card(player.hand, card_name)
        self.discard_pile.append(player.hand[card_index])
        del player.hand[card_index]

    def discard_keeper(self, player_number: int, card_name: str):
        player = self.players[player_number]
        card_index = index_of_card(player.keepers, card_name)

        self.discard_pile.append(player.keepers[card_index])
        del player.keepers[card_index]

        self.check_for_winners()

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

    def play_card_from_hand(self, player_number: int, card_name: str):
        """
        Assumes a validated input (will not check whether the card is actually in the player's hand), but will search the hand for it

        Play a card from a player's hand. Discards it from the player's hand but does NOT add it to the discard pile.

        :arguments:
        - player_number: Index of the player in self.players who is playing the card
        - card_name: The name of the card to play (as a string)
        :return:
        """
        player = self.players[player_number]

        i = index_of_card(player.hand, card_name)
        card_in_hand = player.hand[i]

        del player.hand[i]
        player.cards_played += 1

        self.activate_card(player_number, card_in_hand)

        self.check_for_winners()

    def start_of_turn(self):
        """Apply start of turn effects, including drawing."""
        turn_player = self.players[self.player_turn]

        if self.rule_in_play("no_hand_bonus"):
            if len(turn_player.hand) == 0:
                for i in range(3 + self.inflation()):
                    self.draw(turn_player)

        self.draw_for_turn()

    def is_turn_over(self) -> bool:
        turn_player = self.players[self.player_turn]

        # turn over if out of plays
        plays = self.get_play_rules(self.player_turn)
        if plays <= turn_player.cards_played:
            return True

        # turn over if out of cards in hand + no free actions
        if len(turn_player.hand) == 0:
            return True

        return self.force_turn_over

    def limit_check_player(self, player_number: int):
        player = self.players[player_number]

        keeper_limit = self.get_keeper_limit()
        hand_limit = self.get_hand_limit()

        if keeper_limit is not None:
            for i in range(len(player.keepers) - keeper_limit):
                self.stack.append(GamePhase(GamePhaseType.DISCARD_KEEPER, player_number))

        if hand_limit is not None:
            for i in range(len(player.hand) - hand_limit):
                self.stack.append(GamePhase(GamePhaseType.DISCARD_CARD_FROM_HAND, player_number))

    def get_keeper_limit(self, player_number: Optional[int]=None) -> Optional[int]:
        limit = None
        for rule in self.rules:
            if rule.keeper_limit is not None:
                limit = rule.keeper_limit + self.inflation()

        return limit

    def get_hand_limit(self, player_number: Optional[int]=None) -> Optional[int]:
        limit = None
        for rule in self.rules:
            if rule.hand_limit is not None:
                limit = rule.hand_limit + self.inflation()

        return limit

    def get_keepers_by_name(self, player_number: int) -> list[str]:
        return self.get_all_keepers_by_name()[player_number]

    def get_all_keepers_by_name(self) -> list[list[str]]:
        """
        :return: 2d array indexed by player number. get_all_keepers_by_name[player_number] returns a list of the names of all keepers in play held by said keeper
        """
        return [ [keeper.name for keeper in player.keepers] for player in self.players ]

    def get_cards_in_hand_by_name(self, player_number: int) -> list[str]:
        return [card.name for card in self.players[player_number].hand]

    def get_goals_in_play_by_name(self):
        return [goal.name for goal in self.goals]

    def get_rules_in_play_by_name(self):
        return [rule.name for rule in self.rules]
    # -----------------------------------------------------------
    # PRE REFACTOR CODE
    # -----------------------------------------------------------

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

    def get_available_free_actions(self):
        available_free_actions = []

        for rule in self.rules:
            if rule.free_action and can_use_free_action(self, self.player_turn, rule.name):
                available_free_actions.append(rule.name)

        return available_free_actions

    def play_free_action(self, free_action_name):
        activate_free_action(self, self.player_turn, free_action_name)
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