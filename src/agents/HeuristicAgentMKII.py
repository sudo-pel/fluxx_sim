import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.agents import agent_utils
from src.agents.Agent import Agent
from src.game.cards.card_data import CARD_DATA
from src.game.FluxxEnums import GameConfig, GameState, GamePhaseType
from src.game.utils.general_utils import print_game_state

"""

Implementation plan


"""

@dataclass
class Gameplan:
    """
    goal: goal pertinent to gameplan

    required_cards: set of cards (including goal) required to win via gameplan
    held_cards: set of cards currently held by player, in hand or in play (relevant to goal)
    missing_cards: set of cards required to play goal but not currently held by player
    cards_in_hand: subset of held_cards that are in hand
    cards_in_play: subset of held_cards that are in play

    held_count, missing_count, in_hand_count, in_play_count: all self-explanatory
    goal_in_play: whether goal is in play
    """
    goal: str
    required_cards: set[str]
    held_cards: set[str]
    missing_cards: set[str]
    cards_in_hand: set[str]
    cards_in_play: set[str]
    cards_in_discard: set[str]
    held_count: int
    missing_count: int
    in_hand_count: int
    in_play_count: int
    in_discard_count: int
    goal_in_play: bool

class GameplanSortingOptions(Enum):
    MISSING_COUNT = 0
    IN_HAND_COUNT = 1
    IN_PLAY_COUNT = 2
    HELD_COUNT = 3


ASYMMETRIC_TURN_EXTENDERS = {
    "draw_2_and_use_em",
    "draw_3_play_2_of_them",
    "take_another_turn",
    "todays_special",
}


def rule_options(card_name: str) -> dict:
    """
    Return the RulesOptions dict for a rule card, or empty dict.
    """
    return CARD_DATA[card_name].get("RulesOptions", {})

def card_type(card_name: str) -> str:
    """
    Return the card_type string ('KEEPER', 'GOAL', 'RULE', 'ACTION').
    """
    return CARD_DATA[card_name]["card_type"]

def is_play_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("play") is not None


def is_draw_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("draw") is not None


def is_hand_limit_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("hand_limit") is not None


def is_keeper_limit_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("keeper_limit") is not None


def is_limit_rule(card_name: str) -> bool:
    return is_hand_limit_rule(card_name) or is_keeper_limit_rule(card_name)

class HeuristicAgentMKII(Agent):
    keeper_to_goal: dict[str, list[str]] = defaultdict(list[str])
    for card in CARD_DATA:
        if CARD_DATA[card]["card_type"] == "GOAL":
            for keeper in CARD_DATA[card]["required_keepers"]:
                keeper_to_goal[keeper].append(card)

    def eval_free_action(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        relevant_cards = state.hands[self.player_number] + state.goals + state.keepers[self.player_number]
        gameplans, card_to_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(relevant_cards, state, self.player_number, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        # Check whether there is an immediate route to victory
        for gameplan in gameplans:
            if gameplan.missing_count == 0 and gameplan.in_hand_count <= state.plays_remaining[self.player_number]:
                return priorities # return empty priorities

        for card in cards_to_eval:
            if card == "mystery_play":
                priorities[3].add(card)
            if card == "goal_mill":
                goals_in_hand = [card for card in state.hands[self.player_number] if card_type(card) == "GOAL"]
                for goal in goals_in_hand:
                    gameplan = card_to_gameplan[goal][0]
                    if gameplan.in_discard_count > 0 or gameplan.missing_count > 2:
                        priorities[2].add(card)
                        break
            elif card == "swap_plays_for_draws":
                # If hand is empty then exchange plays for draws = free cards
                if len(state.hands[self.player_number]) == 0:
                    priorities[2].add(card)
                # Otherwise, only consider doing if there are keepers/goals to evaluate
                # TODO: implicit here is the assumption that actions/rules are always worth playing which isnt true, forcing an early turn end can result in not losing the game
                elif gameplans:
                    best_gameplan = gameplans[0]
                    if best_gameplan.in_discard_count > 0 or best_gameplan.missing_count > 2:
                        priorities[2].add(card)
            elif card == "recycling":
                for keeper in state.keepers[self.player_number]:
                    weak_gameplans_only = True
                    card_gameplans = card_to_gameplan[keeper]
                    for gameplan in card_gameplans:
                        if gameplan.in_discard_count == 0 and gameplan.missing_count < 3:
                            weak_gameplans_only = False
                            break
                    if weak_gameplans_only:
                        priorities[1].add(card)
            elif card == "get_on_with_it":
                if state.plays_remaining[self.player_number] == 1:
                    weak_cards_only = True
                    for card_in_hand in state.hands[self.player_number]:
                        if card_type(card_in_hand) == "ACTION":
                            weak_cards_only = False
                            break
                        elif (card_type(card_in_hand) == "KEEPER" or card_type(card_in_hand) == "GOAL") and card_in_hand in card_to_gameplan:
                            for gameplans in card_to_gameplan[card_in_hand]:
                                if gameplans.in_discard_count == 0 and gameplans.missing_count < 3:
                                    weak_cards_only = False
                                    break
                    if weak_cards_only:
                        priorities[1].add(card)
        return priorities

    # TODO: augment this function to take an argument of "additional cards", so that, for example, it can take into account that the card may be being ..
    # TODO: .. played from latent space (currently, card would not be in hand and so gameplan would appear incomplete even if every other card is in hand)
    def eval_play(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, self.player_number, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(state.hands[1 - self.player_number], state, 1 - self.player_number, hand_visible=False, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        plays_remaining = state.plays_remaining[self.player_number]
        cards_drawn = state.cards_drawn[self.player_number]

        # Check whether there is an immediate route to victory
        for gameplan in gameplans:
            if gameplan.missing_count == 0 and gameplan.in_hand_count <= plays_remaining:
                return { 100: {card for card in gameplan.required_cards if (card in cards_to_eval)} }

        if len(state.goals) == 1:
            current_goal_opponent_gameplan, _ = HeuristicAgentMKII.get_gameplans_from_cards(state.goals, state, 1 - self.player_number)
            current_goal_opponent_gameplan = current_goal_opponent_gameplan[0]

        for card in cards_to_eval:
            card_ranked = False

            # used for one of the heuristics below
            if card in card_to_opponent_gameplan:
                opponent_gameplan = card_to_opponent_gameplan[card][0]
                goal_mod = 1 if opponent_gameplan.goal_in_play else 0

            # if the card is a limit card that would force the opponent to discard, strongly consider playing it
            # TODO: take into account forcing yourself to discard gameplan cards (this would be bad)
            if card_type(card) == "RULE" and is_limit_rule(card):
                if is_hand_limit_rule(card):
                    player_discard = max(0, len(state.hands[self.player_number]) - rule_options(card)["hand_limit"])
                    opponent_discard = max(0, len(state.hands[1 - self.player_number]) - rule_options(card)["hand_limit"])
                    priorities[-player_discard - opponent_discard - player_discard].add(card)
                    card_ranked = True
                else:
                    player_discard = max(0, len(state.keepers[self.player_number]) - rule_options(card)["keeper_limit"])
                    opponent_discard = max(0, len(state.keepers[1 - self.player_number]) - rule_options(card)["keeper_limit"])
                    priorities[-player_discard - opponent_discard - player_discard].add(card)
                    card_ranked = True
            # if the opponent is building towards the goal in play, replace it with another goal
            # TODO: make this work alongside "double agenda"
            elif card_type(card) == "GOAL" and len(state.goals) == 1 and current_goal_opponent_gameplan.in_play_count > 1:
                # TODO: take into account discard pile/recursion?
                # only do this if the opponent is not building towards the goal being ranked/they are further from the goal being ranked than the one in play
                if not card_to_opponent_gameplan[card] or card_to_opponent_gameplan[card][0].missing_count > current_goal_opponent_gameplan.missing_count:
                    priorities[6].add(card)
                    card_ranked = True
            # asymmetric turn extenders are good and should be played generally speaking
            elif card in ASYMMETRIC_TURN_EXTENDERS:
                priorities[5].add(card)
                card_ranked = True
            # play a "play" rule if it allows for extension of the current turn
            elif is_play_rule(card):
                if rule_options(card)["play"] > plays_remaining + state.cards_played[self.player_number] + 1:
                    priorities[4].add(card)
                    card_ranked = True
                # if on final play, play the "play" rule if it will reduce the opponents turn
                elif state.plays_remaining[self.player_number] == 1 and rule_options(card)["play"] < state.cards_played[self.player_number] + 1:
                    priorities[4].add(card)
                    card_ranked = True
                else:
                    # otherwise (doesn't extend current turn or reduce opponents turn) don't play it
                    priorities[-1].add(card)
                    card_ranked = True
            # play a "draw" card if it allows for drawing extra cards
            elif is_draw_rule(card) and rule_options(card)["draw"] > cards_drawn:
                priorities[3].add(card)
                card_ranked = True
            elif card_type(card) == "GOAL":
                # under no circumstances play a goal card that will instantly win your opponent the game
                if card in card_to_opponent_gameplan and (opponent_gameplan.missing_cards == {card} or len(opponent_gameplan.missing_cards) == 0):
                    priorities[-100].add(card)
                    card_ranked = True
                # avoid playing goals for which your opponent has a keeper(s) (and where the other required cards aren't in discard)
                elif card in card_to_opponent_gameplan and opponent_gameplan.in_play_count > goal_mod and opponent_gameplan.in_discard_count == 0:
                    priorities[-4 + opponent_gameplan.missing_count].add(card)
                    card_ranked = True
                # avoid playing goal cards that are being worked towards while they are "incomplete" (since they will be overwritten)
                elif card in card_to_gameplan and card_to_gameplan[card][0].in_play_count > 1 and card_to_gameplan[card][0].in_discard_count == 0:
                    priorities[-2].add(card)
                    card_ranked = True
            # play "keeper" cards that are relevant to one another, prioritizing those that are relevant to the current goal
            elif card_type(card) == "KEEPER":
                card_gameplans = card_to_gameplan[card]
                for gameplan in card_gameplans:
                    if gameplan.goal_in_play:
                        goal_mod = 1
                    else:
                        goal_mod = 0
                    priorities[6 - gameplan.missing_count + goal_mod].add(card)
                    card_ranked = True
                    break

            if not card_ranked:
                priorities[0].add(card)

        return priorities

    # TODO: strongly consider linking this with eval_discard_rule_in_play (since you may have the option of discarding rules when running this function)
    def eval_discard(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            # If you can discard an opponent keeper, always try to do so
            # TODO: nuance regarding disallowed keepers
            if card in state.keepers[1 - self.player_number]:
                priorities[2].add(card)
            # If the card pertains to any gameplans with missing pieces not in the discard pile, keep the card around
            # TODO: prioritize cards with stronger gameplans
            elif card in card_to_gameplan:
                card_gameplans = card_to_gameplan[card]
                weak_gameplan = True
                for gameplan in card_gameplans:
                    if gameplan.in_discard_count == 0:
                        weak_gameplan = False
                        break
                if weak_gameplan:
                    priorities[1].add(card)
                else:
                    priorities[-2].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_discard_rule_in_play(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            # Asymmetric turn reduction: if a rule in play lets you play more cards and you've played that many, get rid of the rule so the opponent cannot use it
            if is_play_rule(card):
                if rule_options(card)["play"] == state.cards_played[self.player_number]:
                    priorities[1].add(card)
                else:
                    # Avoid reducing your own turn
                    priorities[-1].add(card)
            # Discard a draw rule if you've drawn that many cards (always will have because draw cards activate at turn start)
            elif is_draw_rule(card):
                priorities[1].add(card)
            # Discard a keeper limit rule if keeping it in play would force you to discard cards.
            elif is_keeper_limit_rule(card):
                if len(state.keepers[self.player_number]) > rule_options(card)["keeper_limit"]:
                    priorities[1].add(card)
                # Don't discard a limit rule if it wouldn't force you to discard but would force your opponent
                elif len(state.keepers[1 - self.player_number]) > rule_options(card)["keeper_limit"]:
                    priorities[-1].add(card)
                else:
                    priorities[0].add(card)
            # Same logic as above, but for hand limit rules
            elif is_hand_limit_rule(card):
                if len(state.hands[self.player_number]) - state.plays_remaining[self.player_number] > rule_options(card)["hand_limit"]:
                    priorities[1].add(card)
                elif len(state.hands[1 - self.player_number]) > rule_options(card)["hand_limit"]:
                    priorities[-1].add(card)
                else:
                    priorities[0].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_add_card_in_play_to_hand(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, self.player_number, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            # Prioritize taking keepers away from the opponent
            if card in state.keepers[1 - self.player_number]:
                priorities[3].add(card)
            # If a limit card is going to force discard, remove it from play
            elif is_hand_limit_rule(card) and len(state.hands[self.player_number]) - state.plays_remaining[self.player_number] > rule_options(card)["hand_limit"]:
                priorities[2].add(card)
            elif is_keeper_limit_rule(card) and len(state.keepers[self.player_number]) > rule_options(card)["keeper_limit"]:
                priorities[2].add(card)
            # If a card is in play that forms an "incomplete gameplan", protect it by returning it to hand
            elif card_type(card) == "GOAL":
                gameplan = card_to_gameplan[card][0]
                if gameplan.in_play_count > 1 and gameplan.in_discard_count == 0:
                    priorities[1].add(card)
                else:
                    priorities[0].add(card)
            else:
                priorities[0].add(card)


        return priorities

    def eval_share_cards_from_latent_space(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, self.player_number, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, 1 - self.player_number, hand_visible=False, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            # If the card constitutes part of a gameplan, take it from the latent space (depending on how possible the gameplan is)
            # ... avoid doing so if the gameplan has cards in the discard pile
            if card in card_to_gameplan:
                for gameplan in card_to_gameplan[card]:
                    if gameplan.in_discard_count == 0:
                        priorities[10-gameplan.missing_count].add(card)
                        break
            elif card in card_to_opponent_gameplan:
                for gameplan in card_to_opponent_gameplan[card]:
                    if gameplan.in_discard_count == 0:
                        priorities[9-gameplan.missing_count].add(card)
                        break
            # duplicates here are fine because all card priorities if added otherwise, are positive
            priorities[0].add(card)

        return priorities

    def eval_give_keeper_to_opponent(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, self.player_number, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKII.get_gameplans_from_cards(cards_to_eval, state, 1 - self.player_number, hand_visible=False, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        # Avoid giving keepers that constitute good gameplans for the player or for the opponent
        for card in cards_to_eval:
            card_ranked = False

            if card in card_to_gameplan:
                for gameplan in card_to_gameplan[card]:
                    if gameplan.in_discard_count == 0:
                        priorities[gameplan.missing_count-10].add(card)
                        card_ranked = True
                        break
            elif card in card_to_opponent_gameplan:
                for gameplan in card_to_opponent_gameplan[card]:
                    if gameplan.in_discard_count == 0:
                        priorities[gameplan.missing_count-9].add(card)
                        card_ranked = True
                        break
            if not card_ranked:
                priorities[0].add(card)

        return priorities

    # TODO: add support for "use free action" (may just leave without though)
    game_phase_to_eval_function = {
        GamePhaseType.PLAY_CARD_FOR_TURN: eval_play,
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE: eval_play,
        GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE: eval_play,

        GamePhaseType.ACTIVATE_FREE_ACTION: eval_free_action,

        GamePhaseType.DISCARD_CARD_FROM_HAND: eval_discard,
        GamePhaseType.DISCARD_KEEPER: eval_discard,
        GamePhaseType.DISCARD_KEEPER_IN_PLAY: eval_discard,
        GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY: eval_discard,
        GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND: eval_discard,
        GamePhaseType.DISCARD_GOAL_IN_PLAY: eval_discard,

        GamePhaseType.DISCARD_RULE_IN_PLAY: eval_discard_rule_in_play,

        GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND: eval_add_card_in_play_to_hand,

        GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND: eval_share_cards_from_latent_space,
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT: eval_share_cards_from_latent_space,
        GamePhaseType.SELECT_KEEPER_TO_STEAL: eval_share_cards_from_latent_space,
        GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE: eval_share_cards_from_latent_space,

        GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE: eval_give_keeper_to_opponent,
    }

    def __init__(self, game_config: GameConfig, player_number: int):
        self.game_config = game_config
        self.player_number = player_number

    @staticmethod
    def get_gameplans_from_cards(cards: list[str], game_state: GameState, player_number: int, hand_visible: bool = True, sort_by: Optional[GameplanSortingOptions] = None, reverse: bool = False) -> tuple[list[Gameplan], dict[str, list[Gameplan]]]:
        """
        Takes a list of cards (strings) and returns:
        - list of Gameplans that can be achieved with any subset of those cards
        - dict mapping each card to a list of Gameplans that can be achieved with that card

        Lists of Gameplans (within both return values) are sorted by in ascending order of missing cards (so the gameplans closest to fruition are first)
        """

        # TODO: unique goals like 5 keepers and special keeper requirements like disallowed keepers and optional keepers
        # When a goal is seen: generate a gameplan corresponding to that goal
        # When a keeper is seen: generate a gameplan corresponding to each goal that the keeper is pertinent to
        # Avoid generating duplicate gameplans by hashing via a tuple of the goal and all required keepers (which uniquely identifies a goal)
        gameplans: dict[tuple[str], Gameplan] = {}
        for card in cards:
            card_type = CARD_DATA[card]["card_type"]
            if card_type not in {"GOAL", "KEEPER"}:
                continue
            if card_type == "GOAL":
                goal_names = [card]
            else:
                goal_names = HeuristicAgentMKII.keeper_to_goal[card]

            for goal_name in goal_names:
                # is a tuple for the sake of hashing
                required_cards = tuple(CARD_DATA[goal_name]["required_keepers"] + [goal_name])
                required_cards_set = set(required_cards)

                if required_cards in gameplans:
                    continue

                cards_in_hand = {card for card in game_state.hands[player_number] if card in required_cards} if hand_visible else set()
                cards_in_play = {card for card in required_cards if (card in game_state.keepers[player_number]) or (card in game_state.goals)}

                held_cards = cards_in_hand | cards_in_play
                missing_cards = required_cards_set - held_cards
                held_count = len(held_cards)
                missing_count = len(missing_cards)
                in_hand_count = len(cards_in_hand)
                in_play_count = len(cards_in_play)
                goal_in_play = card in game_state.goals
                cards_in_discard = {card for card in game_state.discard_pile if (card in required_cards_set)}
                in_discard_count = len(cards_in_discard)

                # TODO: add support for goals with optional or disallowed keepers (must enrich Gameplan datatype)
                gameplan = Gameplan(
                    goal_name,
                    required_cards_set,
                    held_cards,
                    missing_cards,
                    cards_in_hand,
                    cards_in_play,
                    cards_in_discard,
                    held_count,
                    missing_count,
                    in_hand_count,
                    in_play_count,
                    in_discard_count,
                    goal_in_play
                )

                gameplans[required_cards] = gameplan

        card_to_gameplan: dict[str, list[Gameplan]] = defaultdict(list)
        for required_cards, gameplan in gameplans.items():
            for card in required_cards:
                card_to_gameplan[card].append(gameplan)

        gameplan_list = [g for g in gameplans.values()]
        gameplan_list.sort(key=lambda gameplan: gameplan.goal)
        if sort_by is not None:
            if sort_by == GameplanSortingOptions.MISSING_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.missing_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.missing_count, reverse=reverse)
            elif sort_by == GameplanSortingOptions.HELD_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.held_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.held_count, reverse=reverse)
            elif sort_by == GameplanSortingOptions.IN_HAND_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.in_hand_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.in_hand_count)
            elif sort_by == GameplanSortingOptions.IN_PLAY_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.in_play_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.in_play_count, reverse=reverse)

        return gameplan_list, card_to_gameplan

    def act(self, state: GameState):
        current_phase = state.stack[-1]
        action_mask = agent_utils.observe_hot_encoded(self, state, self.game_config)["action_mask"]
        cards_to_choose_from = [self.game_config.card_list[i] for i in range(len(self.game_config.card_list)) if action_mask[i] == 1]
        priorities: dict[int, set[str]] = HeuristicAgentMKII.game_phase_to_eval_function[current_phase.type](self, state, cards_to_choose_from)

        # adding "no free action" if masked in
        if action_mask[-1] == 1:
            priorities[0].add("no_free_action")

        if len(priorities) == 0:
            print_game_state(state)
            raise Exception("No cards to choose from")

        max_priority = max(priorities.keys())
        choice = random.choice(list(priorities[max_priority]))

        if choice == "no_free_action":
            return len(action_mask)-1, [], None
        else:
            return self.game_config.card_list.index(choice), [], None


