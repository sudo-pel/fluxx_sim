from typing import Optional

from src.agents.utils import agent_utils
from src.game import game_messages, GameSchema
from src.game.FluxxEnums import CardType, CardZone, ExtendedCardZone, AnyCardZone, GamePhase, GamePhaseType
from dataclasses import dataclass

@dataclass
class CardLocation:
    """
    Class (kinda like a complex enum) for encoding where cards are for passing between functions
    """
    zone: CardZone
    index: int
    zone_index: Optional[int] = None

def trash_selected_card(game: GameSchema, user_number: int, card_location: CardLocation, add_to_discard: bool):
    if card_location.zone == CardZone.RULES:
        rule_discarded = get_selected_card(game, card_location)
        if rule_discarded.name == "double_agenda" and len(game.goals) == 2:
            game.stack.append(GamePhase(GamePhaseType.DISCARD_GOAL_IN_PLAY, user_number, decisions_left=1))
        if rule_discarded.name == "triple_agenda":
            limit = 1
            if "double_agenda" in game.get_rules_in_play_by_name():
                limit = 2
            if len(game.goals) > limit:
                game.stack.append(GamePhase(GamePhaseType.DISCARD_GOAL_IN_PLAY, user_number, decisions_left=len(game.goals) - limit))
        if add_to_discard:
            game.discard_pile.append(game.rules[card_location.index])
        del game.rules[card_location.index]
        return

    if card_location.zone == CardZone.KEEPERS:
        if add_to_discard:
            game.discard_pile.append(game.players[card_location.zone_index].keepers[card_location.index])
        del game.players[card_location.zone_index].keepers[card_location.index]
        return

    if card_location.zone == CardZone.GOALS:
        if add_to_discard:
            game.discard_pile.append(game.goals[card_location.index])
        del game.goals[card_location.index]
        return

    if card_location.zone == CardZone.DISCARD_PILE:
        del game.discard_pile[card_location.index]
        return

    if card_location.zone == CardZone.HAND:
        if add_to_discard:
            game.discard_pile.append(game.players[card_location[1]].hand[card_location[2]])
        del game.players[card_location[1]].hand[card_location[2]]
        return

    raise Exception("Error: invalid card location")

def get_selected_card(game: GameSchema, card_location: CardLocation):
    if card_location.zone == CardZone.RULES:
        return game.rules[card_location.index]

    if card_location.zone == CardZone.KEEPERS:
        return game.players[card_location.zone_index].keepers[card_location.index]

    if card_location.zone == CardZone.GOALS:
        return game.goals[card_location.index]

    if card_location.zone == CardZone.DISCARD_PILE:
        return game.discard_pile[card_location.index]

    if card_location.zone == CardZone.HAND:
        return game.players[card_location.zone_index].hand[card_location.index]

    raise Exception("Error: invalid card location")

def find_card_in_play_by_name(game: GameSchema, card_name: str) -> Optional[CardLocation]:
    for i, rule in enumerate(game.rules):
        if rule.name == card_name:
            return CardLocation(CardZone.RULES, i)
    for i, player in enumerate(game.players):
        for j, keeper in enumerate(player.keepers):
            if keeper.name == card_name:
                return CardLocation(CardZone.KEEPERS, j, i)
    for i, goal in enumerate(game.goals):
        if goal.name == card_name:
            return CardLocation(CardZone.GOALS, i)
    for i, card in enumerate(game.discard_pile):
        if card.name == card_name:
            return CardLocation(CardZone.DISCARD_PILE, i)
    return None