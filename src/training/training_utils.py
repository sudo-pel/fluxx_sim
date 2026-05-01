# TODO: these are taken from HeuristicAgentMKII; consider keeping the functions only in one place
# Slight difference: optional keepers increase gameplan strength. Disallowed keepers decrease gameplan strength only when in play.
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.game.FluxxEnums import GameState
from src.game.cards.card_data import CARD_DATA


@dataclass
class GameplanExtended:
    """
    goal: goal pertinent to gameplan

    required_cards: set of cards (including goal) required to win via gameplan
    held_cards: set of cards currently held by player, in hand or in play (relevant to goal), not including optionals
    missing_cards: set of cards required to play goal but not currently held by player
    cards_in_hand: subset of held_cards that are in hand
    cards_in_play: subset of held_cards that are in play
    disallowed_in_play: set of cards that are **disallowed** by the goal that are owned by the player and in play
        n.b that disallowed cards being in the hand does not influence chances of victory (although they should not ...
            ... be played in this case, hence the omission)
    held_optional_cards: set of a set of cards which is the union of each corresponding optional_cards_in_hand and optional_cards_in_play
    optional_cards_in_hand: set of a set of cards. Each subset contains cards in the corresponding subset of ...
        optional cards that are in the player's hand.
    optional_cards_in_play: set of a set of cards. Each subset contains cards in the corresponding subset of ...
        optional cards that are in the player's hand.
    optional_cards_in_discard: see above logic and translate accordingly
    missing_optional_cards: set of a set of cards. Each subset contains cards in the corresponding subset of ...
        optional cards that are missing.
    missing_optional_subgaol_count: number of unfulfilled optional keeper subsets
        n.b: optional keepers specify ONE OF some set of cards (so, not strictly optional). One of these sets is ...
            ... considered a "subgoal". Fulfilling a goal card with optional cards requires having all of the ...
            ... required keepers in play while simultaneously fulfilling all subgoals.

    held_count, missing_count, in_hand_count, in_play_count, disallowed_in_play_count: all self-explanatory
    goal_in_play: whether goal is in play
    """
    goal: str
    required_cards: set[str]
    held_cards: set[str]
    missing_cards: set[str]
    cards_in_hand: set[str]
    cards_in_play: set[str]
    cards_in_discard: set[str]
    disallowed_cards_in_play: set[str]
    optional_cards: list[set[str]]
    held_optional_cards: list[set[str]]
    optional_cards_in_hand: list[set[str]]
    optional_cards_in_play: list[set[str]]
    optional_cards_in_discard: list[set[str]]
    missing_optional_cards: list[set[str]]
    held_count: int
    missing_count: int
    in_hand_count: int
    in_play_count: int
    in_discard_count: int
    disallowed_cards_in_play_count: int
    missing_optional_subgoal_count: int
    goal_in_play: bool
    score: float

class GameplanExtendedSortingOptions(Enum):
    MISSING_COUNT = 0
    IN_HAND_COUNT = 1
    IN_PLAY_COUNT = 2
    HELD_COUNT = 3
    GAMEPLAN_SCORE = 4

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

keeper_to_goal: dict[str, list[str]] = defaultdict(list[str])
for card in CARD_DATA:
    if CARD_DATA[card]["card_type"] == "GOAL":
        for keeper in CARD_DATA[card]["required_keepers"]:
            keeper_to_goal[keeper].append(card)

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
    gameplans: dict[tuple[str], GameplanExtended] = {}
    for card in cards:
        card_type = CARD_DATA[card]["card_type"]
        if card_type not in {"GOAL", "KEEPER"}:
            continue
        if card_type == "GOAL":
            goal_names = [card]
        else:
            goal_names = keeper_to_goal[card]

        for goal_name in goal_names:
            # is a tuple for the sake of hashing
            required_cards = tuple(CARD_DATA[goal_name]["required_keepers"] + [goal_name])
            required_cards_set = set(required_cards)
            disallowed_cards = set(CARD_DATA[goal_name]["disallowed_keepers"]) # no need to hash: goal name and ...
            # ... required keepers uniquely specifies a gameplan
            optional_cards = [set(cards) for cards in CARD_DATA[goal_name]["optional_keepers"]]


            if required_cards in gameplans:
                continue

            cards_in_hand = {card for card in game_state.hands[player_number] if card in required_cards} if hand_visible else set()
            cards_in_play = {card for card in required_cards if (card in game_state.keepers[player_number]) or (card in game_state.goals)}
            disallowed_cards_in_play = {card for card in game_state.keepers[player_number] if card in disallowed_cards}

            optional_cards_in_hand = [{card for card in game_state.hands[player_number] if card in subset} for subset in optional_cards]
            optional_cards_in_play = [{card for card in game_state.keepers[player_number] if card in subset} for subset in optional_cards]
            optional_cards_in_discard = [{card for card in game_state.discard_pile if card in subset} for subset in optional_cards]

            held_cards = cards_in_hand | cards_in_play
            held_optional_cards = [in_hand | in_play for in_hand, in_play in zip(optional_cards_in_hand, optional_cards_in_play)]
            missing_cards = required_cards_set - held_cards
            missing_optional_cards = [optionals - held for optionals, held in zip(optional_cards, held_optional_cards)]



            held_count = len(held_cards)
            missing_count = len(missing_cards)
            in_hand_count = len(cards_in_hand)
            in_play_count = len(cards_in_play)
            goal_in_play = card in game_state.goals
            cards_in_discard = {card for card in game_state.discard_pile if (card in required_cards_set)}
            in_discard_count = len(cards_in_discard)
            disallowed_cards_in_play_count = len(disallowed_cards_in_play)
            missing_optional_subgoal_count = len({s for s in held_optional_cards if len(s) == 0})

            # TODO: add support for goals with optional or disallowed keepers (must enrich Gameplan datatype)
            gameplan = GameplanExtended(
                goal_name,
                required_cards_set,
                held_cards,
                missing_cards,
                cards_in_hand,
                cards_in_play,
                cards_in_discard,
                disallowed_cards_in_play,
                optional_cards,
                held_optional_cards,
                optional_cards_in_hand,
                optional_cards_in_play,
                optional_cards_in_discard,
                missing_optional_cards,
                held_count,
                missing_count,
                in_hand_count,
                in_play_count,
                in_discard_count,
                disallowed_cards_in_play_count,
                missing_optional_subgoal_count,
                goal_in_play,
                0.0 # see next line
            )

            gameplan.score = score_gameplan(gameplan)
            gameplans[required_cards] = gameplan

    card_to_gameplan: dict[str, list[GameplanExtended]] = defaultdict(list)
    for required_cards, gameplan in gameplans.items():
        for card in required_cards:
            card_to_gameplan[card].append(gameplan)

    gameplan_list = [g for g in gameplans.values()]
    gameplan_list.sort(key=lambda gameplan: gameplan.goal)
    if sort_by is not None:
        if sort_by == GameplanExtendedSortingOptions.MISSING_COUNT:
            gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.missing_count, reverse=reverse)
            for _, l in card_to_gameplan.items():
                l.sort(key=lambda gameplan: gameplan.missing_count, reverse=reverse)
        elif sort_by == GameplanExtendedSortingOptions.HELD_COUNT:
            gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.held_count, reverse=reverse)
            for _, l in card_to_gameplan.items():
                l.sort(key=lambda gameplan: gameplan.held_count, reverse=reverse)
        elif sort_by == GameplanExtendedSortingOptions.IN_HAND_COUNT:
            gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.in_hand_count, reverse=reverse)
            for _, l in card_to_gameplan.items():
                l.sort(key=lambda gameplan: gameplan.in_hand_count)
        elif sort_by == GameplanExtendedSortingOptions.IN_PLAY_COUNT:
            gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.in_play_count, reverse=reverse)
            for _, l in card_to_gameplan.items():
                l.sort(key=lambda gameplan: gameplan.in_play_count, reverse=reverse)
        elif sort_by == GameplanExtendedSortingOptions.GAMEPLAN_SCORE:
            gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.score, reverse=reverse)
            for _, l in card_to_gameplan.items():
                l.sort(key=lambda gameplan: gameplan.score, reverse=reverse)

    return gameplan_list, card_to_gameplan

def score_gameplan(gameplan: GameplanExtended) -> float:
    """
    Gives a gameplan a "score" to be used in reward shaping. Scoring metrics:

    A perfect gameplan (all required cards held) is 10 points. Points are subtracted as follows:
    • -2 for each missing card
    • -2/len(s) for each missing optional card in optional subset s
    • -2 for each disallowed card in play

    (Currently), no point are awarded for chasing the goal currently in play
    """
    score = 10
    score -= gameplan.missing_count * 2
    score -= gameplan.disallowed_cards_in_play_count * 2
    for i, subset in enumerate(gameplan.missing_optional_cards):
        if len(subset) > 0:
            score -= 2 / len(gameplan.optional_cards)

    return score

