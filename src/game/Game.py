import copy
import random
from typing import Optional
from collections import Counter

from src.env.Logger import Logger
from src.game.cards.Card import Card, Rule, Goal, Keeper, Action
from src.game.FluxxEnums import CardType, GamePhase, GamePhaseType, GameState, CardZone, GamePhaseHistory

from src.game.Player import Player
from src.game import game_messages
from src.game.cards import action_cards
from src.game.cards.free_actions import can_use_free_action, activate_free_action
from src.game.GameSchema import GameSchema
from src.game.game_messages import GameMessageType
from src.game.utils.card_effect_utils import find_card_in_play_by_name, get_selected_card, trash_selected_card, \
    CardLocation
from src.game.utils.general_utils import index_of_card


class Game(GameSchema):
    def __init__(self, player_count: int, card_list: list[str], disable_game_messages: bool = False, force_game_state: Optional[GameState] = None, logger: Optional[Logger] = None):
        GameSchema.__init__(self, player_count, card_list, disable_game_messages, force_game_state, logger)

    # there is a class for game state although it is not currently used.
    def get_game_state(self) -> GameState:
        return GameState(
            self.turn_count,
            self.player_count,
            [self.get_cards_in_hand_by_name(i) for i in range(self.player_count)],
            [self.get_keepers_by_name(i) for i in range(self.player_count)],
            self.get_goals_in_play_by_name(),
            self.get_discard_pile_by_name(),
            self.get_draw_pile_by_name(),
            self.get_rules_in_play_by_name(),
            self.stack,
            self.get_available_free_actions(),
            self.winner is not None,
            starting_player = None,
            cards_played = [player.cards_played for player in self.players],
            plays_remaining = [max(self.get_play_rules(i) - player.cards_played, 0) for i, player in enumerate(self.players)],
            cards_drawn = [player.cards_drawn for player in self.players],
        )

    def game_message(self, message: str, message_type: GameMessageType):
        if self.logger:
            self.logger.game_message(message, message_type)

        if self.disable_game_messages:
            return

        if message_type == GameMessageType.SPECIAL_EFFECT:
            game_messages.special_effect(message)
        elif message_type == GameMessageType.NOTIFICATION:
            game_messages.notification(message)
        elif message_type == GameMessageType.GAME_OVER:
            game_messages.game_over(message)
        elif message_type == GameMessageType.DRAWN_CARD:
            game_messages.drawn_card(message)
        elif message_type == GameMessageType.TURN_START:
            game_messages.turn_start(message)


    def add_player_turn_to_stack(self, skip_free_action: bool = False):
        turn_player = self.players[self.player_turn]

        # if free action was skipped, then there will already be a POST_PLAY_CARD_FOR_TURN on the stack from the previous call to add_player_turn_to_stack, that added ACTIVATE_FREE_ACTION in the first place!
        if not skip_free_action:
            self.stack.append(GamePhase(GamePhaseType.POST_PLAY_CARD_FOR_TURN, self.player_turn))
        if len(self.get_available_free_actions()) > 0 and not skip_free_action:
            self.stack.append(GamePhase(GamePhaseType.ACTIVATE_FREE_ACTION, self.player_turn, decisions_left=1))
        else:
            if self.rule_in_play("first_play_random") and len(turn_player.hand) > 0:
                if turn_player.cards_played == 0 and self.get_play_rules(self.player_turn) > 1:
                    card_index = random.randint(0, len(turn_player.hand) - 1)
                    card_name = turn_player.hand[card_index].name
                    self.game_message(f"<< (First Play Random) played {card_name} >>", GameMessageType.SPECIAL_EFFECT)
                    self.play_card_from_hand(self.player_turn, card_name)
                    return

            plays_left = self.get_play_rules(self.player_turn) - self.players[self.player_turn].cards_played

            # Immediately goto end of turn if player has no cards left to play, and does not want to play a free action
            if len(turn_player.hand) > 0 and plays_left > 0:
                self.stack.append(GamePhase(GamePhaseType.PLAY_CARD_FOR_TURN, self.player_turn, decisions_left=plays_left))
                assert self.player_turn == self.stack[-1].acting_player

    def reset(self):
        super().reset()
        for player in self.players:
            for i in range(3):
                self.draw(player)
        self.start_of_turn()
        self.add_player_turn_to_stack()

    def assert_nonempty_stack(self):
        assert len(self.stack) > 0, (
            f"Stack unexpectedly empty. "
            f"Turn: {self.player_turn}, "
            f"Hand: {[c.name for c in self.players[self.player_turn].hand]}, "
            f"Cards played: {self.players[self.player_turn].cards_played}, "
            f"Plays allowed: {self.get_play_rules(self.player_turn)}, "
            f"Force turn over: {self.force_turn_over}, "
            f"Rules: {self.get_rules_in_play_by_name()}"
        )

    def check_current_phase(self) -> GamePhase:
        self.assert_nonempty_stack()
        return self.stack[-1]

    def get_current_phase(self) -> GamePhase:
        """
        The same as check_current_phase, but pops from the phase stack
        """

        self.assert_nonempty_stack()
        return self.stack.pop()

    def is_action_valid(self, phase: GamePhase, card_name: str) -> tuple[bool, Optional[str]]:
        """
        Given a GamePhase and a game state (implicit in the self-reference), validates a given action.

        All actions in the game refer to card names.
        """

        # Specific action validation
        if phase.type == GamePhaseType.PLAY_CARD_FOR_TURN:
            # is it this player's turn?
            if self.player_turn != phase.acting_player: return False, "Not this player's turn"

            # is the card in this player's hard?
            if card_name not in self.get_cards_in_hand_by_name(phase.acting_player): return False, "Card to be played not in hand"

        elif phase.type == GamePhaseType.DISCARD_CARD_FROM_HAND:
            # is the card in this player's hand?
            if card_name not in self.get_cards_in_hand_by_name(phase.acting_player): return False, "Card to be discarded not in hand"

        elif phase.type == GamePhaseType.DISCARD_KEEPER:
            # does the player own this keeper?
            if card_name not in self.get_keepers_by_name(phase.acting_player): return False, "Keeper to be discarded not owned"

        elif phase.type == GamePhaseType.DISCARD_RULE_IN_PLAY:
            if card_name not in self.get_rules_in_play_by_name(): return False, "Rule to be discarded not in play"

        elif phase.type == GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND:
            if card_name not in self.get_cards_in_play_by_name(): return False, "Card to be zapped not in play"

        elif phase.type == GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE:
            if card_name not in self.get_discard_pile_by_name(): return False, "Card to be played not in discard pile"
            if self.discard_pile[index_of_card(self.discard_pile, card_name)].card_type not in [CardType.ACTION, CardType.RULE]: return False, "Card to be played not an action or rule"

        elif phase.type == GamePhaseType.DISCARD_KEEPER_IN_PLAY:
            if card_name not in self.get_all_keepers_by_name_flat(): return False, "Keeper to be discarded not in play"

        elif phase.type == GamePhaseType.SELECT_KEEPER_TO_STEAL:
            if card_name not in self.get_all_keepers_by_name()[phase.acting_player ^ 1]: return False, "Keeper to be stolen not owned (by opponent)"

        elif phase.type == GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE:
            if card_name not in self.get_all_keepers_by_name()[phase.acting_player ^ 1]: return False, "Keeper to be exchanged not owned (by opponent)"

        elif phase.type == GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE:
            if card_name not in self.get_all_keepers_by_name()[phase.acting_player]: return False, "Keeper to be exchanged not owned (by player)"
            if card_name == phase.labelled_card.name: return False, "Keeper to be exchanged cannot be exchanged for itself"

        elif phase.type == GamePhaseType.ACTIVATE_FREE_ACTION:
            if card_name != "no_free_action" and card_name not in self.get_available_free_actions(): return False, "Free action to be activated not available"

        elif phase.type == GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY:
            if card_name not in self.get_keepers_by_name(phase.acting_player): return False, "Keeper to be discarded not owned (by player)"

        elif phase.type == GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND:
            if card_name != "no_free_action":
                card_index = index_of_card(self.players[phase.acting_player].hand, card_name)
                if card_index == -1: return False, "Card to be discarded not in player hand"
                if self.players[phase.acting_player].hand[card_index].card_type not in phase.card_types: return False, "Card to be discarded not of the correct type"

        elif phase.type == GamePhaseType.DISCARD_GOAL_IN_PLAY:
            if card_name not in self.get_goals_in_play_by_name(): return False, "Goal to be discarded not in play"

        elif phase.type.contains_latent_space():
            if index_of_card(phase.latent_space, card_name) == -1: return False, "Card to be played not in latent space"

        return True, None

    def assert_card_conservation(self):
        total = (
                len(self.draw_pile)
                + len(self.discard_pile)
                + sum(len(p.hand) for p in self.players)
                + sum(len(p.keepers) for p in self.players)
                + len(self.goals)
                + len(self.rules)
                + sum(
            len(phase.latent_space)
            for phase in self.stack
            if hasattr(phase, 'latent_space') and phase.latent_space is not None
        )
                + sum(
            1 for phase in self.stack
            if hasattr(phase, 'card') and phase.card is not None
        )
        )
        """
        assert total == len(self.deck), (
            f"Card conservation violated: expected {len(self.deck)}, found {total}. "
            f"Draw: {len(self.draw_pile)}, "
            f"Discard: {len(self.discard_pile)}, "
            f"Hands: {[len(p.hand) for p in self.players]}, "
            f"Keepers: {[len(p.keepers) for p in self.players]}, "
            f"Goals: {len(self.goals)}, "
            f"Rules: {len(self.rules)}, "
            f"Stack latent: {sum(len(ph.latent_space) for ph in self.stack if hasattr(ph, 'latent_space') and ph.latent_space is not None)}, "
            f"Stack cards: {sum(1 for ph in self.stack if hasattr(ph, 'card') and ph.card is not None)}"
        )
        """

    # We note that the simulator will receive an integer and then decode it into something more complex for the game simulator to consume
    def step(self, card_name: str):
        if self.winner is not None:
            if card_name is not None:
                # this should never happen
                raise Exception("Game already won")
            return

        current_phase = self.get_current_phase()
        acting_player = current_phase.acting_player

        action_valid, error_message = self.is_action_valid(current_phase, card_name)
        if not action_valid:
            raise Exception(f"Invalid action: {error_message}")

        if current_phase.type == GamePhaseType.PLAY_CARD_FOR_TURN:
            self.play_card_from_hand(acting_player, card_name)

        elif current_phase.type == GamePhaseType.DISCARD_CARD_FROM_HAND:
            self.discard_card(acting_player, card_name)
            if current_phase.decisions_left > 1 and len(self.players[acting_player].hand) > 0:
                self.stack.append(GamePhase(GamePhaseType.DISCARD_CARD_FROM_HAND, acting_player, decisions_left=current_phase.decisions_left - 1))

        elif current_phase.type == GamePhaseType.DISCARD_KEEPER:
            self.discard_keeper(acting_player, card_name)
            if current_phase.decisions_left > 1 and len(self.players[acting_player].keepers) > 0:
                self.stack.append(GamePhase(GamePhaseType.DISCARD_KEEPER, acting_player, decisions_left=current_phase.decisions_left - 1))

        elif current_phase.type == GamePhaseType.DISCARD_RULE_IN_PLAY:
            card_index = index_of_card(self.rules, card_name)
            card_loc = CardLocation(CardZone.RULES, card_index)
            trash_selected_card(self, acting_player, card_loc, True)

        elif current_phase.type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE:
            card_index = index_of_card(current_phase.latent_space, card_name)
            card_to_play = current_phase.latent_space[card_index]
            del current_phase.latent_space[card_index]
            if current_phase.decisions_left > 1 and len(current_phase.latent_space) > 0:
                current_phase.decisions_left -= 1
                self.stack.append(current_phase)
            else:
                for card in current_phase.latent_space:
                    self.stack.append(GamePhase(GamePhaseType.DEFERRED_ADD_CARD_TO_DISCARD_PILE, acting_player, card=card))
            self.activate_card(acting_player, card_to_play)

        elif current_phase.type == GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND:
            # Fluxx decks have no duplicates, so can just try zapping the card from all locations
            card_location = find_card_in_play_by_name(self, card_name)
            card = get_selected_card(self, card_location)
            trash_selected_card(self, acting_player, card_location, False)
            self.players[acting_player].hand.append(card)

        elif current_phase.type == GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND:
            card_index = index_of_card(current_phase.latent_space, card_name)
            card_to_draw = current_phase.latent_space[card_index]
            del current_phase.latent_space[card_index]
            if current_phase.decisions_left > 1 and len(current_phase.latent_space) > 0:
                current_phase.decisions_left -= 1
                self.stack.append(current_phase)
            else:
                self.players[acting_player ^ 1].hand.extend(current_phase.latent_space)
            self.players[acting_player].hand.append(card_to_draw)

        elif current_phase.type == GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE:
            card_index = index_of_card(self.discard_pile, card_name)
            card_to_play = self.discard_pile[card_index]
            del self.discard_pile[card_index]
            self.activate_card(acting_player, card_to_play)

        elif current_phase.type == GamePhaseType.DISCARD_KEEPER_IN_PLAY:
            card_location = find_card_in_play_by_name(self, card_name)
            trash_selected_card(self, acting_player, card_location, True)

        elif current_phase.type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT:
            card_index = index_of_card(current_phase.latent_space, card_name)
            card_to_play = current_phase.latent_space[card_index]
            del current_phase.latent_space[card_index]
            if current_phase.decisions_left > 1 and len(current_phase.latent_space) > 0:
                current_phase.decisions_left -= 1
                self.stack.append(current_phase)
            else:
                for card in current_phase.latent_space:
                    # For safety, activation of each card is deferred until resolution of the prior one (hence the use of latent space here)
                    self.stack.append(
                        GamePhase(GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE, acting_player ^ 1, latent_space=[card], decisions_left=1) # acting player is taken to be the opponent
                    )

            self.activate_card(acting_player, card_to_play)

        elif current_phase.type == GamePhaseType.SELECT_KEEPER_TO_STEAL:
            card_location = find_card_in_play_by_name(self, card_name)
            card = get_selected_card(self, card_location)
            trash_selected_card(self, acting_player, card_location, False)
            self.players[acting_player].keepers.append(card)

        elif current_phase.type == GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE:
            card_index = index_of_card(self.players[acting_player ^ 1].keepers, card_name)
            card = self.players[acting_player ^ 1].keepers[card_index]
            del self.players[acting_player ^ 1].keepers[card_index]
            self.players[acting_player].keepers.append(card)

            self.stack.append(GamePhase(GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE, acting_player, decisions_left=1, labelled_card=card))

        elif current_phase.type == GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE:
            card_index = index_of_card(self.players[acting_player].keepers, card_name)
            card = self.players[acting_player].keepers[card_index]
            del self.players[acting_player].keepers[card_index]
            self.players[acting_player ^ 1].keepers.append(card)

        elif current_phase.type == GamePhaseType.ACTIVATE_FREE_ACTION:
            if card_name != "no_free_action":
                self.play_free_action(card_name)
            else:
                self.add_player_turn_to_stack(skip_free_action=True)

        elif current_phase.type == GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY:
            card_index = index_of_card(self.players[acting_player].keepers, card_name)
            card = self.players[acting_player].keepers[card_index]
            del self.players[acting_player].keepers[card_index]
            self.discard_pile.append(card)

        elif current_phase.type == GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND:
            if card_name == "no_free_action":
                for i in range(current_phase.counter):
                    self.draw(self.players[acting_player])
            else:
                card_index = index_of_card(self.players[acting_player].hand, card_name)
                card = self.players[acting_player].hand[card_index]
                del self.players[acting_player].hand[card_index]
                self.discard_pile.append(card)

                self.stack.append(GamePhase(GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND, acting_player, decisions_left=0, card_types=current_phase.card_types, counter=current_phase.counter + 1, on_complete=current_phase.on_complete))

        elif current_phase.type == GamePhaseType.DISCARD_GOAL_IN_PLAY:
            card_index = index_of_card(self.goals, card_name)
            card = self.goals[card_index]
            del self.goals[card_index]
            self.discard_pile.append(card)

        # early stop if the game is over
        if self.winner is not None:
            return

        # ---
        # ACTIONLESS PHASES (some actions need to be deferred after prompts)
        # ---

        while self.check_current_phase().type.is_actionless():
            current_phase = self.get_current_phase()
            acting_player = current_phase.acting_player

            if current_phase.type == GamePhaseType.POST_PLAY_CARD_FOR_TURN:
                self.handle_turn_over()

            elif current_phase.type == GamePhaseType.TURN_END:
                self.end_of_turn()
                self.start_of_turn()
                self.add_player_turn_to_stack()

            elif current_phase.type == GamePhaseType.DEFERRED_ADD_CARD_TO_DISCARD_PILE:
                self.discard_pile.append(current_phase.card)

            elif current_phase.type == GamePhaseType.DEFERRED_DRAW_CARD:
                for i in range(current_phase.decisions_left):
                    self.draw(self.players[acting_player])

            elif current_phase.type == GamePhaseType.DEFERRED_PLAY_GOAL:
                self.goals.append(current_phase.card)

        # ---
        # CORRECTNESS ASSERTION
        # ---
        # After each step, the total number of cards in all zones should be constant
        self.assert_card_conservation()

        next_phase = self.check_current_phase()
        if next_phase.type == GamePhaseType.PLAY_CARD_FOR_TURN:
            player = self.players[next_phase.acting_player]
            assert len(player.hand) > 0, (
                f"PLAY_CARD_FOR_TURN with empty hand. "
                f"acting_player: {next_phase.acting_player}, "
                f"player_turn: {self.player_turn}, "
                f"cards_played: {player.cards_played}, "
                f"cards_drawn: {player.cards_drawn}, "
                f"plays_allowed: {self.get_play_rules(next_phase.acting_player)}, "
                f"draw_rules: {self.get_draw_rules(next_phase.acting_player)}, "
                f"draw_pile: {len(self.draw_pile)}, "
                f"discard: {len(self.discard_pile)}, "
                f"stack: {[(p.type.name, p.acting_player) for p in self.stack]}, "
                f"force_turn_over: {self.force_turn_over}, "
                f"rules: {self.get_rules_in_play_by_name()}, "
                f"free_actions_played: {self.played_free_actions}"
            )

    def handle_turn_over(self):
        """
        Checks whether the turn player's turn is over, advancing the turn if so and adding another PLAY+POST_CARD_FOR_TURN to the stack otherwise
        """
        # apply draw effects in case
        self.draw_for_turn()

        if self.is_turn_over():
            self.handle_end_of_turn()
        else:
            # Allow the player to play another card
            self.add_player_turn_to_stack()

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
        self.force_turn_over = False
        self.played_free_actions = set()

        if not self.extra_turn:

            old_player = self.player_turn
            for phase in self.stack:
                assert not (phase.type == GamePhaseType.PLAY_CARD_FOR_TURN and phase.acting_player == old_player), (
                    f"Stale PLAY_CARD_FOR_TURN for player {old_player} still on stack at turn end. "
                    f"Stack: {[(p.type.name, p.acting_player) for p in self.stack]}"
                )

            self.extra_turns_taken = 0
            self.game_message(f"<< End of player {self.player_turn} turn >>", GameMessageType.SPECIAL_EFFECT)
            self.player_turn = (self.player_turn + 1) % self.player_count

        else:
            self.extra_turn = False
            self.game_message(f"<< Player {self.player_turn} extra turn >>", GameMessageType.SPECIAL_EFFECT)

        self.turn_count += 1

        # game logging: cleanest to put at the end of each turn
        if self.logger is not None:
            self.logger.game_stepped(self.get_game_state())

    def discard_card(self, player_number: int, card_name: str):
        """
        Discard a card from a player's hand. Does not check whether the card is actually in the player's hand.
        """
        player = self.players[player_number]

        card_index = index_of_card(player.hand, card_name)

        self.game_message(f"<< Player {player_number} discarded {player.hand[card_index].name} >>", GameMessageType.NOTIFICATION)

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
        if self.rule_in_play("double_agenda"):
            if len(self.goals) == 2:
                self.stack.append(GamePhase(GamePhaseType.DEFERRED_PLAY_GOAL, player_number, decisions_left=1, card=goal))
                self.stack.append(GamePhase(GamePhaseType.DISCARD_GOAL_IN_PLAY, player_number, decisions_left=1))
            else:
                self.goals.append(goal)
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
        self.stack.append(GamePhase(GamePhaseType.DEFERRED_ADD_CARD_TO_DISCARD_PILE, player_number, card=action))
        action_cards.activate_action(action.name, self, player_number)

    def activate_card(self, player_number: int, card_to_play):
        """Activate a card. Is the result of 'playing a card', but is not the same thing as it"""

        self.game_message(f"<< Player {player_number} plays {card_to_play.name} >>", GameMessageType.SPECIAL_EFFECT)

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

        Play a card from a player's hand. Does NOT add it to the discard pile

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

        turn_player.cards_played = 0
        turn_player.cards_drawn = 0

        if self.rule_in_play("no_hand_bonus"):
            if len(turn_player.hand) == 0:
                for i in range(3 + self.inflation()):
                    self.draw(turn_player)

        assert turn_player.cards_drawn == 0, (
            f"cards_drawn not reset: {turn_player.cards_drawn} for player {self.player_turn}"
        )

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
            if len(player.keepers) - keeper_limit > 0:
                self.stack.append(GamePhase(GamePhaseType.DISCARD_KEEPER, player_number, decisions_left=len(player.keepers) - keeper_limit))

        if hand_limit is not None:
            if len(player.hand) - hand_limit > 0:
                self.stack.append(GamePhase(GamePhaseType.DISCARD_CARD_FROM_HAND, player_number, decisions_left=len(player.hand) - hand_limit))

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

    def get_draw_pile_by_name(self) -> list[str]:
        return [card.name for card in self.draw_pile]

    def get_discard_pile_by_name(self) -> list[str]:
        return [card.name for card in self.discard_pile]

    def get_keepers_by_name(self, player_number: int) -> list[str]:
        return self.get_all_keepers_by_name()[player_number]

    def get_all_keepers_by_name_flat(self) -> list[str]:
        return [k for players in self.get_all_keepers_by_name() for k in players]

    def get_all_keepers_by_name(self) -> list[list[str]]:
        """
        :return: 2d array indexed by player number. get_all_keepers_by_name[player_number] returns a list of the names of all keepers in play held by said keeper
        """
        return [ [keeper.name for keeper in player.keepers] for player in self.players ]

    def get_cards_in_hand(self, player_number: int) -> list[Card]:
        return self.players[player_number].hand

    def get_cards_in_hand_by_name(self, player_number: int) -> list[str]:
        return [card.name for card in self.players[player_number].hand]

    def get_goals_in_play_by_name(self):
        return [goal.name for goal in self.goals]

    def get_rules_in_play_by_name(self):
        return [rule.name for rule in self.rules]

    def get_cards_in_play_by_name(self):
        cards_in_play = []
        for player in self.players:
            cards_in_play.extend(player.keepers)
        cards_in_play.extend(self.goals)
        cards_in_play.extend(self.rules)
        return [card.name for card in cards_in_play]

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
            play += 1 + self.inflation()

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
            if card_played[option] is not None:
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

        if self.winner is not None:
            return

        winners = []

        for current_goal in self.goals:
            # TODO: special goal cards (cards in hand, etc)
            if current_goal.name == "5_keepers":
                keeper_counts = [len(p.keepers) for p in self.players]
                for i, player in enumerate(self.players):
                    keeper_count = len(player.keepers)
                    if keeper_count >= 5 + self.inflation():
                        if keeper_count == max(keeper_counts) and Counter(keeper_counts)[keeper_count] == 1:
                            winners.append(player)
                continue
            elif current_goal.name == "10_cards_in_hand":
                hand_counts = [len(p.hand) for p in self.players]
                for i, player in enumerate(self.players):
                    hand_count = len(player.hand)
                    if hand_count >= 10 + self.inflation():
                        if hand_count == max(hand_counts) and Counter(hand_counts)[hand_count] == 1:
                            winners.append(player)
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

                    if optional_keepers < len(current_goal.optional_keepers) + self.inflation():
                        win_cancelled = True

                    if not win_cancelled:
                        winners.append(player)

        if len(winners) == 1:
            self.winner = winners[0].id
            if self.logger is not None:
                self.logger.game_over(self.winner, self.get_game_state())

    def shuffle_discard_pile_into_draw(self):
        random.shuffle(self.discard_pile)
        self.draw_pile = self.discard_pile
        self.discard_pile = []

    def get_card_from_draw_pile(self) -> Optional[Card]:
        """
        Get a card from the draw pile. Not the same as drawing a card.
        If both the draw pile and discard pile are empty, do not draw a card.
        """
        if not self.draw_pile:
            self.shuffle_discard_pile_into_draw()

        if not self.draw_pile:
            return None

        card_drawn = self.draw_pile.pop()

        self.game_message(f"[[ DRAW '{card_drawn.name}' ]]", GameMessageType.DRAWN_CARD)

        if not self.draw_pile:
            self.shuffle_discard_pile_into_draw()

        return card_drawn

    def draw(self, player: Player):
        """Draw a card from the deck and add it to the hand of player 'player'. Does NOT increment player.cards_drawn"""
        self.game_message(f"player_{player.id} drawing", GameMessageType.DRAWN_CARD)
        card_drawn = self.get_card_from_draw_pile()

        if card_drawn is None and len(self.draw_pile) > 0:
            raise Exception("No card drawn even though there are cards in the draw pile")

        if card_drawn is None:
            return

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

    def get_available_free_actions(self) -> list[str]:
        available_free_actions = []

        for rule in self.rules:
            if rule.free_action and can_use_free_action(self, self.player_turn, rule.name):
                available_free_actions.append(rule.name)

        return available_free_actions

    def play_free_action(self, free_action_name):
        self.played_free_actions.add(free_action_name)
        self.game_message(f"<< Player {self.player_turn} plays free action '{free_action_name}' >>", GameMessageType.SPECIAL_EFFECT)
        activate_free_action(self, self.player_turn, free_action_name)

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

    def inflation(self) -> int:
        """
        Returns 1 or 0 depending on whether rule card "inflation" is in play.
        :return:
        """
        if self.rule_in_play("inflation"):
            #game_messages.notification("(Inflation bonus)")
            return 1
        else:
            return 0